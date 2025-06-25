from __future__ import annotations

"""Celery application object.
Настраивает брокер/бекенд Redis и расписание задач.
"""

import os
import ssl
import logging
from urllib.parse import urlparse
from celery import Celery
from celery.signals import worker_process_init
from src.config.settings import settings

logger = logging.getLogger(__name__)

# Переменные окружения с разумными значениями по умолчанию для локального запуска
redis_url = settings.REDIS_URL

# Определяем, используется ли защищенное соединение
parsed_url = urlparse(redis_url)
is_redis_ssl = parsed_url.scheme == 'rediss'

logger.info(f"Redis URL scheme: {parsed_url.scheme}, SSL enabled: {is_redis_ssl}")

# Настройки для SSL соединения
broker_transport_options = {}
result_backend_transport_options = {}

if is_redis_ssl:
    # Настройки SSL для брокера и результата
    # Получаем ssl_cert_reqs из настроек
    ssl_cert_reqs_map = {
        'CERT_NONE': ssl.CERT_NONE,
        'CERT_OPTIONAL': ssl.CERT_OPTIONAL,
        'CERT_REQUIRED': ssl.CERT_REQUIRED,
    }
    cert_reqs = ssl_cert_reqs_map.get(settings.REDIS_SSL_CERT_REQS, ssl.CERT_NONE)
    
    ssl_config = {
        'ssl_cert_reqs': cert_reqs,
        'ssl_ca_certs': None,
        'ssl_certfile': None,
        'ssl_keyfile': None,
        'ssl_check_hostname': settings.REDIS_SSL_CHECK_HOSTNAME,
    }
    
    broker_transport_options = ssl_config
    result_backend_transport_options = ssl_config
    
    logger.info(f"Configured Redis SSL with cert_reqs={settings.REDIS_SSL_CERT_REQS}, check_hostname={settings.REDIS_SSL_CHECK_HOSTNAME}")
else:
    logger.info("Using non-SSL Redis connection")

celery_app = Celery(
    "sidelined",
    broker=redis_url,
    backend=redis_url,
    include=[
        "src.tasks.master_update",  # регистрируем модуль с задачами
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
    "create-autoyaps": {
        "task": "src.tasks.create_autoyaps.create_autoyaps",
        "schedule": settings.AUTOYAPS_SYNC_PERIOD_SECONDS,
        "options": {"queue": "default"},
    },
    "master-update": {
        "task": "src.tasks.master_update.master_update",
        "schedule": settings.MASTER_UPDATE_PERIOD_SECONDS,
        "options": {"queue": "master_update"},
    },
}

# Роутинг задач
celery_app.conf.task_routes = {
    "src.tasks.create_autoyaps.create_autoyaps": {"queue": "default"},
    "src.tasks.master_update.master_update": {"queue": "master_update"},
}

# Дополнительные настройки для стабильной работы
celery_app.conf.update(
    # Настройки worker'ов
    worker_prefetch_multiplier=1,  # Уменьшаем prefetch для равномерного распределения
    task_acks_late=False,  # Подтверждаем задачи только после успешного выполнения
    worker_max_tasks_per_child=10,  # Перезапускаем worker каждые 10 задач
    
    # Настройки сериализации
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    
    # Настройки времени выполнения
    task_soft_time_limit=60*50,  # 50 минут мягкий лимит
    task_time_limit=60*55,       # 55 минут жесткий лимит
    
    # SSL настройки для Redis
    broker_transport_options=broker_transport_options,
    result_backend_transport_options=result_backend_transport_options,
) 