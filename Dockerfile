# استخدم صورة Python 3.9 الرسمية
FROM python:3.9-slim

# إعداد متغيرات البيئة لتحسين الأداء
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# تعيين مجلد العمل
WORKDIR /app

# نسخ ملف المتطلبات وتثبيت الحزم
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# نسخ باقي الملفات
COPY . /app/

# بدء التطبيق
CMD ["python", "train.py"]
