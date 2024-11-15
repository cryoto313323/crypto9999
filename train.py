import logging
import os

import hydra
from omegaconf import DictConfig
from models import MODELS
from data_loader import get_dataset
from factory.trainer import Trainer
from factory.evaluator import Evaluator
from factory.profit_calculator import ProfitCalculator
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from path_definition import HYDRA_PATH

from utils.reporter import Reporter
from data_loader.creator import create_dataset, preprocess
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import logging
from omegaconf import DictConfig
from functools import partial
from flask import Flask
from threading import Thread
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger



import yaml
from datetime import datetime, timedelta

import requests
import csv
import os
import pytz
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)


# إعداد تفاصيل API
url = "https://api.binance.com/api/v3/klines"

symbols  = [ "ALGOUSDT"];


current_date = datetime.now()

data = {
    "window_size": 5,
    "train_start_date": "2020-01-01 13:30:00",  # تاريخ ثابت
    "train_end_date": (current_date - timedelta(days=5)).strftime("%Y-%m-%d 09:30:00"),
    "valid_start_date": (current_date - timedelta(days=5)).strftime("%Y-%m-%d 10:30:00"),
    "valid_end_date": (current_date + timedelta(days=2)).strftime("%Y-%m-%d 10:30:00"),
    "features": "Date, open, High, Low, close, volume",
    "indicators_names": "rsi macd"
}

# كتابة البيانات إلى ملف YAML بدون علامات التنصيص
with open("configs/hydra/dataset_loader/common.yaml", "w") as file:
    yaml.dump(data, file, default_flow_style=False, allow_unicode=True, sort_keys=False)



def check_and_delete_file(filename):
    try:
        # فتح الملف والتحقق من آخر سطر
        with open(filename, 'r') as file:
            lines = file.readlines()
            if not lines:
                print(f"No data in file {filename}.")
                os.remove(filename)
                return None, False

            last_line = lines[-1]  # قراءة آخر سطر في الملف
            last_date = datetime.strptime(last_line.split(',')[0], '%Y-%m-%d %H:%M:%S%z')
            
            # إذا كانت البيانات غير كاملة حتى تاريخ الأمس
            if last_date.date() < (datetime.now(pytz.utc).date() - timedelta(days=1)):
                os.remove(filename)  # حذف الملف إذا كانت البيانات قديمة
                print(f"File {filename} deleted because it doesn't cover up to yesterday.")
                return None, False  # إرجاع None مع False للإشارة إلى أن البيانات ناقصة

            # إذا كانت البيانات كاملة حتى تاريخ الأمس، حفظ قيمة الإغلاق (close)
            yesterday_close = float(last_line.split(',')[5])  # قيمة الإغلاق
            return yesterday_close, True  # إرجاع قيمة الإغلاق وTrue للدلالة على أن البيانات كاملة

    except Exception as e:
        print(f"An error occurred while checking file {filename}: {e}")
        return None, False  # في حالة حدوث خطأ، إرجاع None وFalse

def add_future_dates(filename, symbol):
    today = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    future_dates = [today + timedelta(days=i) for i in range(3)]  # today and the next two days

    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        for future_date in future_dates:
            formatted_date = future_date.strftime('%Y-%m-%d 00:00:00+00:00')
            writer.writerow([formatted_date, symbol,'1','1','1','1','1'])


data_folder = '/opt/render/project/src/data'

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

data_filename = os.path.join(data_folder,'data1.csv')

def fetch_and_save_data(symbol, start_date, end_date):
    url = "http://crypto1212.great-site.net/crypto.php"  # Add API URL here
    params = {
        'symbol': symbol,
        'interval': '1d',
        'startTime': int(start_date.timestamp() * 1000),
        'endTime': int(end_date.timestamp() * 1000)
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        # التحقق من وجود أخطاء في الرد
        if isinstance(data, dict) and 'code' in data:
            print(f"Error fetching data: {data['msg']}")
            return False

        # التحقق من وجود بيانات
        if not data:
            print(f"No data available for {symbol} from {start_date.date()} to {end_date.date()}")
            return False

        # كتابة البيانات إلى ملف CSV
        with open(data_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            for entry in data:
                timestamp = datetime.fromtimestamp(entry[0] / 1000, tz=pytz.utc).strftime('%Y-%m-%d %H:%M:%S%z')
                writer.writerow([
                    timestamp, symbol, entry[1], entry[2], entry[3], entry[4], entry[5]
                ])
        print(f"Data for {symbol} from {start_date.date()} to {end_date.date()} saved successfully.")
        return True

    except requests.RequestException as e:
        print(f"Request error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def train(cfg: DictConfig):
    start_date = datetime(2020, 1, 1, tzinfo=pytz.utc)
    end_date = datetime.now(pytz.utc) - timedelta(days=1)
    period = timedelta(days=90)

    title = ''
    increase_threshold = -0.03
    saved_percentage=0
  
    for symbol in symbols:
        # //
        if os.path.exists(data_filename):
            os.remove(data_filename)  # Delete the file if it exists

        # Write headers only when creating the file
        with open(data_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'])

        current_start = start_date
        data_available = False

        while current_start < end_date:
            current_end = min(current_start + period, end_date)
            print(f"Fetching data for {symbol} from {current_start.date()} to {current_end.date()}")
        
            # Fetch data for each period
            if fetch_and_save_data(symbol, current_start, current_end):
                data_available = True  # Data fetched successfully for at least one period

            current_start = current_end + timedelta(days=1)
    
        if not data_available:
            print(f"Skipping {symbol} due to insufficient data.")
            continue  # Skip symbol if no data is available

   
        yesterday_close, data_complete = check_and_delete_file(data_filename)
        if not data_complete:
            continue
        # Add three future dates at the end of the file with the symbol

        add_future_dates(data_filename, symbol)

        print("Data download complete for all symbols with data from the beginning of 2020.")
        # //
        if cfg.load_path is None and cfg.model is None:
            msg = 'either specify a load_path or config a model.'
            logger.error(msg)
            raise Exception(msg)

        elif cfg.load_path is not None:
            dataset_ = pd.read_csv(cfg.load_path)
            if 'Date' not in dataset_.keys():
                dataset_.rename(columns={'timestamp': 'Date'}, inplace=True)
            if 'High' not in dataset_.keys():
                dataset_.rename(columns={'high': 'High'}, inplace=True)
            if 'Low' not in dataset_.keys():
                dataset_.rename(columns={'low': 'Low'}, inplace=True)

            dataset, profit_calculator = preprocess(dataset_, cfg, logger)

        elif cfg.model is not None:
            dataset, profit_calculator = get_dataset(cfg.dataset_loader.name, cfg.dataset_loader.train_start_date,
                                cfg.dataset_loader.valid_end_date, cfg)

        cfg.save_dir = os.getcwd()
        reporter = Reporter(cfg)
        reporter.setup_saving_dirs(cfg.save_dir)
        model = MODELS[cfg.model.type](cfg.model)

        dataset_for_profit = dataset.copy()
        dataset_for_profit.drop(['prediction'], axis=1, inplace=True)
        dataset.drop(['predicted_high', 'predicted_low'], axis=1, inplace=True)
     
        if cfg.validation_method == 'simple':
            train_dataset = dataset[
                (dataset['Date'] > cfg.dataset_loader.train_start_date) & (
                            dataset['Date'] < cfg.dataset_loader.train_end_date)]
            valid_dataset = dataset[
                (dataset['Date'] > cfg.dataset_loader.valid_start_date) & (
                            dataset['Date'] < cfg.dataset_loader.valid_end_date)]
            Trainer(cfg, train_dataset, None, model).train()
            mean_prediction = Evaluator(cfg, test_dataset=valid_dataset, model=model, reporter=reporter).evaluate()
          
        elif cfg.validation_method == 'cross_validation':
            n_split = 3
            tscv = TimeSeriesSplit(n_splits=n_split)

            for train_index, test_index in tscv.split(dataset):
                train_dataset, valid_dataset = dataset.iloc[train_index], dataset.iloc[test_index]
                Trainer(cfg, train_dataset, None, model).train()
                mean_prediction = Evaluator(cfg, test_dataset=valid_dataset, model=model, reporter=reporter).evaluate()

            reporter.add_average()
        
        x = ProfitCalculator(cfg, dataset_for_profit, profit_calculator, mean_prediction, reporter).profit_calculator()
        predicted_high = x[0]['predicted_high'].iloc[0]
        predicted_low = x[0]['predicted_low'].iloc[0]
        predicted_mean = x[0]['predicted_mean'].iloc[0]
        predicted_high_formated="{:.18f}".format(predicted_high)
        predicted_low_formated="{:.18f}".format(predicted_low)
        predicted_mean_formated="{:.18f}".format(predicted_mean)
        increase = (predicted_mean - yesterday_close) / yesterday_close
        if increase > increase_threshold:
           saved_percentage = increase * 100
        else:
            continue 
        title += f'رمز العملة: {symbol}\n'
        title += f'نسبة الزيادة المتوقعة: {round(saved_percentage, 1)}%\n'
        title += f'اعلى سعر متوقع لليوم⬆️:\n {predicted_high_formated}\n'
        title += f'اقل سعر متوقع لليوم⬇️:\n {predicted_low_formated}\n'
        title += f'سعر الإغلاق المتوقع لليوم:\n {predicted_mean}\n'
        title += '---\n'
        print('..............................d')
        print(yesterday_close)
        reporter.print_pretty_metrics(logger)
        reporter.save_metrics()

    # title += 'لا تجعل التنبؤات محور تداولك. ركز على التحليل العميق وإدارة المخاطر، واستند إلى البيانات والحقائق لاتخاذ قرارات مستنيرة.\n'
    print(title)

    return title  # Return the title or any other relevant data



app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

TOKEN = '7247002552:AAFfzqoRJ95XmwOLDB6Pn2etQTSCU3zT4Pc'

AUTHORIZED_USERS = [895650332,1796556765,991558864]#,1796556765

async def data(update: Update, context: ContextTypes.DEFAULT_TYPE, cfg: DictConfig) -> None:
    # الحصول على النتيجة الكبيرة
    result = train(cfg)

    # تقسيم النص عند الفاصل ---
    parts = result.split('---')

    # إرسال كل جزء من الأجزاء بشكل منفصل
    for part in parts:
        # التأكد من أن الجزء ليس فارغًا قبل إرساله
        if part.strip():  # تأكد من أن الجزء ليس فقط مسافات فارغة
            await update.message.reply_text(part.strip())  # إرفاق أي مسافات غير ضرورية


async def check_authorized_user(update: Update) -> bool:
    user_id = update.message.from_user.id
    if user_id in AUTHORIZED_USERS:
        return True
    return False

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await check_authorized_user(update):
        await update.message.reply_text('ليس لديك صلاحية للوصول إلى هذا البوت.')
        return
    
    keyboard = [[KeyboardButton("توقع")]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)
    await update.message.reply_text('مرحبًا! اضغط على الزر لتوقع النتيجة.', reply_markup=reply_markup)

async def handle_prediction(update: Update, context: ContextTypes.DEFAULT_TYPE, cfg: DictConfig) -> None:
    if not await check_authorized_user(update):
        await update.message.reply_text('ليس لديك صلاحية للوصول إلى هذا البوت.')
        return
    
    if update.message.text == "توقع":
        await data(update, context, cfg)

async def daily_prediction(cfg: DictConfig, application: Application) -> None:
    for user_id in AUTHORIZED_USERS:
        try:
            # الحصول على النتيجة من دالة train
            result = train(cfg)

            # تقسيم الرسالة بناءً على الفاصل ---
            parts = result.split('---')

            # إرسال كل جزء من الأجزاء بشكل منفصل
            for part in parts:
                # التأكد من أن الجزء ليس فارغًا قبل إرساله
                if part.strip():  # التأكد من عدم إرسال جزء فارغ
                    await application.bot.send_message(user_id, part.strip())

        except Exception as e:
            print(f"فشل في إرسال التوقع إلى {user_id}: {e}")



@hydra.main(config_path=HYDRA_PATH, config_name="train")
def main(cfg: DictConfig) -> None:
    application = Application.builder().token(TOKEN).build()
    scheduler = AsyncIOScheduler()
    trigger = CronTrigger(hour=7, minute=0, second=0, timezone="Asia/Baghdad")
    scheduler.add_job(daily_prediction, trigger, args=[cfg, application])
    scheduler.start()
    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, partial(handle_prediction, cfg=cfg)))  # تمرير cfg هنا
    application.run_polling()

if __name__ == '__main__':
    flask_thread = Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 8080})
    flask_thread.start()
    main()


