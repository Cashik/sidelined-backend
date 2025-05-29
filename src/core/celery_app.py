from __future__ import annotations

"""Celery application object.
Настраивает брокер/бекенд Redis и расписание задач.
"""

import os
from celery import Celery
from celery.signals import worker_process_init
from src.config.settings import settings

# Переменные окружения с разумными значениями по умолчанию для локального запуска
redis_url = settings.REDIS_URL

celery_app = Celery(
    "sidelined",
    broker=redis_url,
    backend=redis_url,
    include=[
        "src.tasks.sync_posts",  # регистрируем модуль с задачами
        "src.tasks.cleanup_posts",  # регистрируем модуль с задачами очистки
        "src.tasks.create_autoyaps",  # регистрируем модуль с задачами создания авто-постов
    ],
)

@worker_process_init.connect
def init_worker(**kwargs):
    """
    Инициализация worker процесса.
    Сбрасываем соединения базы данных при форке процесса.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Импортируем функцию сброса только когда она нужна
        from src.database import reset_celery_engine
        reset_celery_engine()
        logger.info("Celery worker process initialized with fresh database connections")
    except Exception as e:
        logger.error(f"Failed to initialize worker process: {e}")

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
    "create-autoyaps": {
        "task": "src.tasks.create_autoyaps.create_autoyaps",
        "schedule": settings.AUTOYAPS_SYNC_PERIOD_SECONDS,
        "options": {"queue": "default"},
    },
}

# Роутинг задач
celery_app.conf.task_routes = {
    "src.tasks.sync_posts.sync_posts": {"queue": "default"},
    "src.tasks.cleanup_posts.cleanup_old_posts": {"queue": "default"},
    "src.tasks.create_autoyaps.create_autoyaps": {"queue": "default"},
}

# Дополнительные настройки для стабильной работы
celery_app.conf.update(
    # Настройки worker'ов
    worker_prefetch_multiplier=1,  # Уменьшаем prefetch для равномерного распределения
    task_acks_late=True,  # Подтверждаем задачи только после успешного выполнения
    worker_max_tasks_per_child=1000,  # Перезапускаем worker каждые 1000 задач
    
    # Настройки сериализации
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    
    # Настройки времени выполнения
    task_soft_time_limit=300,  # 5 минут мягкий лимит
    task_time_limit=600,       # 10 минут жесткий лимит
) 