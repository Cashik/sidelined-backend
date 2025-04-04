FROM python:3.12-alpine

# Устанавливаем необходимые системные зависимости
RUN apk add --no-cache \
    postgresql-libs \
    gcc \
    musl-dev \
    postgresql-dev \
    curl

WORKDIR /root/app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=$PYTHONPATH:./src

# Копируем только requirements.txt сначала для лучшего кэширования
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальные файлы проекта
COPY . .

# Очищаем ненужные зависимости для сборки
RUN apk del gcc musl-dev postgresql-dev

# Проверяем, что все миграции применены и запускаем приложение
CMD alembic upgrade head && uvicorn src.main:app --host 0.0.0.0 --port 8000