import json

"""
Базовые утилиты для всего проекта.

Тут должны быть только те утилиты, которые не связанны с бизнес логикой.
"""

import time
def now_timestamp():
    """Получение текущего timestamp в секундах"""
    return int(time.time())


def format_promo_code(code: str) -> str:
    """Форматирование промо-кода"""
    return code.strip().lower()
