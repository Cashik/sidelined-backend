from datetime import datetime, timedelta
from enum import Enum
from uuid import uuid4
from typing import Optional, List, Dict, ClassVar
from sqlalchemy import Enum
from dataclasses import dataclass
from decimal import Decimal
import time
from src import enums, schemas

"""
tokens - токены overseer
coins - монеты в нутри приложения
"""

# Функция для получения текущего timestamp в секундах
def now_timestamp():
    return int(time.time())

# Функция для получения timestamp 50 часов назад
def past_timestamp(hours=50):
    return int(time.time() - hours * 3600)

