import logging
import os
import hydra
from omegaconf import DictConfig
import requests
import csv
from datetime import datetime, timedelta
import pytz
import pandas as pd
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from functools import partial
from flask import Flask
from threading import Thread
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from path_definition import HYDRA_PATH
from utils.reporter import Reporter
from data_loader.creator import create_dataset, preprocess

# إعدادات Telegram
TOKEN = '7247002552:AAFfzqoRJ95XmwOLDB6Pn2etQTSCU3zT4Pc'
AUTHORIZED_USERS = [895650332,1796556765,991558864]  # إضافة أو تعديل المستخدمين المصرح لهم

# إعدادات logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# إعداد تفاصيل API
url = "https://api.binance.com/api/v3/klines"
symbols = ["ALGOUSDT"]
data_folder = '/opt/render/project/src/data'

# التأكد من وجود مجلد البيانات
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

data_filename = os.path.join(data_folder, 'data1.csv')

def fetch_and_save_data(symbol, start_date, end_date):
    params = {
        'symbol': symbol,
        'interval': '1d',
        'startTime': int(start_date.timestamp() * 1000),
        'endTime': int(end_date.timestamp() * 1000)
    }

    response = requests.get(url, params=params)
    data = response.json()

    if isinstance(data, dict) and 'code' in data and data['code'] == -1121:
        logger.error(f"Error fetching data for {symbol}.")
        return False
    
    # التأكد من توفر البيانات
    if not data:
        logger.warning(f"No data available for {symbol} from {start_date.date()} to {end_date.date()}")
        return False  # لا توجد بيانات

    # كتابة البيانات في ملف CSV
    with open(data_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        for entry in data:
            timestamp = datetime.fromtimestamp(entry[0] / 1000, tz=pytz.utc).strftime('%Y-%m-%d 00:00:00+00:00')
            writer.writerow([timestamp, symbol, entry[1], entry[2], entry[3], entry[4], entry[5]])

    logger.info(f"Data for {symbol} saved successfully.")
    return True  # تم جلب البيانات بنجاح

def check_and_delete_file(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            if not lines:
                os.remove(filename)
                return None, False

            last_line = lines[-1]
            last_date = datetime.strptime(last_line.split(',')[0], '%Y-%m-%d %H:%M:%S%z')
            if last_date.date() < (datetime.now(pytz.utc).date() - timedelta(days=1)):
                os.remove(filename)
                return None, False

            yesterday_close = float(last_line.split(',')[5])  # قيمة الإغلاق
            return yesterday_close, True

    except Exception as e:
        logger.error(f"Error while checking file {filename}: {e}")
        return None, False

def train(cfg: DictConfig):
    start_date = datetime(2020, 1, 1, tzinfo=pytz.utc)
    end_date = datetime.now(pytz.utc) - timedelta(days=1)
    period = timedelta(days=90)

    title = ''
    increase_threshold = -0.03
    saved_percentage = 0
  
    for symbol in symbols:
        if os.path.exists(data_filename):
            os.remove(data_filename)  # حذف الملف إذا كان موجودًا

        # كتابة العناوين فقط عند إنشاء الملف لأول مرة
        with open(data_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'])

        current_start = start_date
        data_available = False

        while current_start < end_date:
            current_end = min(current_start + period, end_date)
            logger.info(f"Fetching data for {symbol} from {current_start.date()} to {current_end.date()}")

            if fetch_and_save_data(symbol, current_start, current_end):
                data_available = True

            current_start = current_end + timedelta(days=1)

        if not data_available:
            logger.warning(f"Skipping {symbol} due to insufficient data.")
            continue  # Skip symbol if no data is available

        yesterday_close, data_complete = check_and_delete_file(data_filename)
        if not data_complete:
            continue

        logger.info("Data download complete for all symbols with data from the beginning of 2020.")

    return title  # Return the title or any other relevant data

@app.route('/start', methods=['GET'])
def start():
    return "مرحباً بالعالم!"

# تشغيل التطبيق Flask في Thread
flask_thread = Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 8080})
flask_thread.start()

# إعداد وإطلاق Telegram Bot مع APScheduler
@hydra.main(config_path=HYDRA_PATH, config_name="train")
def main(cfg: DictConfig) -> None:
    application = Application.builder().token(TOKEN).build()
    scheduler = AsyncIOScheduler()
    trigger = CronTrigger(hour=7, minute=0, second=0, timezone="Asia/Baghdad")
    scheduler.add_job(daily_prediction, trigger, args=[cfg, application])
    scheduler.start()
    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, partial(handle_prediction, cfg=cfg))) 
    application.run_polling()

if __name__ == '__main__':
    main()
