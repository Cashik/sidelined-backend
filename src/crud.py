from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from sqlmodel import Session, select, delete, or_
import random
from sqlalchemy import func
import time

from src import models, schemas, enums, exceptions
from src.config import settings

import logging

logger = logging.getLogger(__name__)


async def get_or_create_user(address: str, chain_id: int, session: Session) -> models.User:
    user = session.exec(select(models.User).where(models.User.address == address, models.User.chain_id == chain_id)).first()
    if user:
            return user
    else:
        user = models.User(address=address, chain_id=chain_id)
        session.add(user)
        session.commit()
        session.refresh(user)
        return user


async def get_user_by_id(user_id: int, session: Session) -> Optional[models.User]:
    """
    Получает пользователя по id
    
    Args:
        user_id (int): ID пользователя
        session (Session): Сессия базы данных
        
    Returns:
        Optional[models.User]: Объект пользователя или None, если пользователь не найден
    """
    return session.exec(select(models.User).where(models.User.id == user_id)).first()
