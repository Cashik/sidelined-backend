from __future__ import annotations

"""Celery application object.
Настраивает брокер/бекенд Redis и расписание задач.
"""

import os
from celery import Celery
from src.config.settings import settings

# Переменные окружения с разумными значениями по умолчанию для локального запуска
REDIS_HOST = settings.REDIS_HOST
REDIS_PORT = settings.REDIS_PORT
REDIS_PASSWORD = settings.REDIS_PASSWORD

# redis://[:password@]host:port/db
if REDIS_PASSWORD:
    redis_url = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"
else:
    redis_url = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"

celery_app = Celery(
    "sidelined",
    broker=redis_url,
    backend=redis_url,
    include=[
        "src.tasks.sync_posts",  # регистрируем модуль с задачами
        "src.tasks.cleanup_posts",  # регистрируем модуль с задачами очистки
    ],
)

# ---- Celery конфигурация ----
# Интервал синхронизации постов (сек). По умолчанию 60 сек для dev.
celery_app.conf.beat_schedule = {
    "sync-posts": {
        "task": "src.tasks.sync_posts.sync_posts",
        "schedule": settings.POST_SYNC_PERIOD_SECONDS,
        "options": {"queue": "default"},
    },
    "cleanup-old-posts": {
        "task": "src.tasks.cleanup_posts.cleanup_old_posts",
        "schedule": settings.POST_CLEANUP_TIME_SECONDS,
        "options": {"queue": "default"},
    },
}

# Роутинг задач
celery_app.conf.task_routes = {
    "src.tasks.sync_posts.sync_posts": {"queue": "default"},
    "src.tasks.cleanup_posts.cleanup_old_posts": {"queue": "default"},
} 