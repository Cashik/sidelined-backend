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
    try:
        # Twitter формат: 'Fri May 30 08:13:53 +0000 2025'
        return int(datetime.strptime(date, "%a %b %d %H:%M:%S %z %Y").timestamp())
    except Exception:
        # fallback на старый способ, если вдруг формат другой
        from email.utils import parsedate_to_datetime
        return int(parsedate_to_datetime(date).timestamp())

def timestamp_to_X_date(timestamp: int) -> str:
    """
    Преобразование timestamp в формат X (Twitter)
    YYYY-MM-DDTHH:mm:ssZ. The oldest UTC timestamp from which the Posts will be provided. Timestamp is in second granularity and is inclusive (i.e. 12:00:01 includes the first second of the minute).
    """
    
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%dT%H:%M:%SZ')


def streak_to_multiplier(streak: int) -> float:
    # рассчитывает мультипликатор за серию
    if streak == 0:
        return 1
    elif streak == 1:
        return 1.25
    else:
        return 1.5
    

def loyalty_to_multiplier(user_loyalty: int) -> float:
    # рассчитывает мультипликатор за лояльность
    loyalty_to_bonus = {
        10: 1.125,
        20: 1.25,
        30: 1.35,
        40: 1.5,
        50: 1.75,
        60: 2,
        70: 2.25,
        80: 2.5,
        90: 2.75,
        100: 3
    }
    max_multiplier = 1
    for loyalty, multiplier in loyalty_to_bonus.items():
        if user_loyalty >= loyalty:
            max_multiplier = max(max_multiplier, multiplier)
    return max_multiplier
    