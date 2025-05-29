import json
from email.utils import parsedate_to_datetime
import time
from datetime import datetime

"""
Базовые утилиты для всего проекта.

Тут должны быть только те утилиты, которые не связанны с бизнес логикой.
"""

def now_timestamp():
    """Получение текущего timestamp в секундах"""
    return int(time.time())


def format_promo_code(code: str) -> str:
    """Форматирование промо-кода"""
    return code.strip().lower()


def parse_date_to_timestamp(date: str) -> int:
    return int(parsedate_to_datetime(date).timestamp())

def timestamp_to_X_date(timestamp: int) -> str:
    """
    Преобразование timestamp в формат X (Twitter)
    YYYY-MM-DDTHH:mm:ssZ. The oldest UTC timestamp from which the Posts will be provided. Timestamp is in second granularity and is inclusive (i.e. 12:00:01 includes the first second of the minute).
    """
    
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%dT%H:%M:%SZ')
