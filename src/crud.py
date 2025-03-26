from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict
from sqlalchemy import select, delete, or_, and_, func, desc
from sqlalchemy.orm import Session
import random
import time

from src import models, schemas, enums, exceptions
from src.config import settings

import logging

logger = logging.getLogger(__name__)


async def get_or_create_user(address: str, chain_id: int, session: Session) -> models.User:
    stmt = select(models.User).where(
        models.User.address == address,
        models.User.chain_id == chain_id
    )
    user = session.execute(stmt).scalar_one_or_none()
    
    if user:
        return user
    else:
        user = models.User(address=address, chain_id=chain_id)
        session.add(user)
        session.commit()
        session.refresh(user)
        return user


async def get_user_by_id(user_id: int, session: Session) -> Optional[models.User]:
    stmt = select(models.User).where(models.User.id == user_id)
    return session.execute(stmt).scalar_one_or_none()


async def get_user_chats_summary(db: Session, user_id: int) -> List[schemas.ChatSummary]:
    """
    Получение списка чатов пользователя с краткой информацией:
    - id чата
    - название чата
    - время последнего изменения
    """
    # Получаем чаты пользователя, которые не удалены (visible=True)
    chats_stmt = select(models.Chat).where(
        models.Chat.user_id == user_id,
        models.Chat.visible == True
    ).order_by(desc(models.Chat.created_at))
    
    chats = db.execute(chats_stmt).scalars().all()
    
    # Получаем время последнего сообщения для каждого чата
    result = []
    for chat in chats:
        # Находим последнее сообщение для определения времени обновления
        last_message_stmt = select(models.Message).where(
            models.Message.chat_id == chat.id
        ).order_by(desc(models.Message.created_at)).limit(1)
        
        last_message = db.execute(last_message_stmt).scalar_one_or_none()
        
        # Используем время последнего сообщения или время создания чата, если сообщений нет
        last_updated_at = last_message.created_at if last_message else chat.created_at
        
        # Создаем объект ChatSummary
        result.append(schemas.ChatSummary(
            id=chat.id,
            title=chat.title,
            last_updated_at=last_updated_at
        ))
    
    return result


async def get_user_chat(db: Session, chat_id: int, user_id: int, from_nonce: Optional[int] = None, to_nonce: Optional[int] = None) -> schemas.Chat:
    """
    Получение информации о чате пользователя с сообщениями в диапазоне nonce
    from_nonce - с какого nonce начать (включительно)
    to_nonce - до какого nonce включать (включительно)
    Если from_nonce=None, начинает с начала чата
    Если to_nonce=None, включает до последнего сообщения
    если чат не найден или не принадлежит пользователю, то ошибка
    """
    # Получаем чат
    chat_stmt = select(models.Chat).where(
        models.Chat.id == chat_id,
        models.Chat.visible == True
    )
    chat = db.execute(chat_stmt).scalar_one_or_none()
    
    # Проверяем, существует ли чат
    if not chat:
        raise exceptions.ChatNotFoundException()
    
    # Проверяем, принадлежит ли чат пользователю
    if chat.user_id != user_id:
        raise exceptions.UserNotChatOwnerException()
    
    # Формируем запрос для получения сообщений с учетом nonce
    messages_stmt = select(models.Message).where(
        models.Message.chat_id == chat_id
    )
    
    # Добавляем условия для from_nonce и to_nonce, если они указаны
    if from_nonce is not None:
        messages_stmt = messages_stmt.where(models.Message.nonce >= from_nonce)
    
    if to_nonce is not None:
        messages_stmt = messages_stmt.where(models.Message.nonce <= to_nonce)
    
    # Сортируем сообщения по nonce и selected_at
    messages_stmt = messages_stmt.order_by(
        models.Message.nonce, 
        desc(models.Message.selected_at)
    )
    
    messages = db.execute(messages_stmt).scalars().all()
    
    # Группируем сообщения по nonce, отбирая последнее выбранное для каждого nonce
    messages_dict: Dict[int, List[schemas.Message]] = {}
    
    for message in messages:
        # Преобразуем модель в схему
        schema_message = schemas.Message(
            content=message.content,
            sender=message.sender,
            recipient=message.recipient,
            model=message.model,
            nonce=message.nonce,
            created_at=message.created_at,
            selected_at=message.selected_at
        )
        
        # Группируем сообщения по nonce
        if message.nonce not in messages_dict:
            messages_dict[message.nonce] = []
        
        messages_dict[message.nonce].append(schema_message)
    
    # Создаем объект Chat
    return schemas.Chat(
        id=chat.id,
        title=chat.title,
        messages=messages_dict
    )


async def add_message(db: Session, chat_id: Optional[int], message: schemas.Message, user_id: int) -> schemas.Chat:
    """
    Добавление сообщения в чат
    Если chat_id не указан, создается новый чат
    Если nonce указан в сообщении, то все последующие сообщения удаляются
    Если nonce указан и в чате есть сообщение с таким nonce, но роль не юзерская, то ошибка
    Если nonce не указан, то добавляется в конец чата
    """
    # Если chat_id не указан, создаем новый чат
    if chat_id is None:
        # Генерируем заголовок чата из первых слов сообщения
        title = message.content.strip()[:30] + "..." if len(message.content) > 30 else message.content
        
        # Создаем новый чат
        new_chat = models.Chat(
            user_id=user_id,
            title=title,
            created_at=message.created_at  # Используем время создания сообщения
        )
        
        db.add(new_chat)
        db.commit()
        db.refresh(new_chat)
        
        chat_id = new_chat.id
    else:
        # Проверяем существование чата и права доступа
        chat_stmt = select(models.Chat).where(
            models.Chat.id == chat_id,
            models.Chat.visible == True
        )
        chat = db.execute(chat_stmt).scalar_one_or_none()
        
        if not chat:
            raise exceptions.ChatNotFoundException()
        
        if chat.user_id != user_id:
            raise exceptions.UserNotChatOwnerException()
    
    # Проверяем, если nonce уже существует и это сообщение ассистента
    existing_message_stmt = select(models.Message).where(
        models.Message.chat_id == chat_id,
        models.Message.nonce == message.nonce
    )
    existing_message = db.execute(existing_message_stmt).scalar_one_or_none()
    
    if existing_message and existing_message.sender == enums.Role.ASSISTANT and message.sender == enums.Role.USER:
        raise exceptions.InvalidNonceException()
    
    # Удаляем все сообщения с большим nonce
    delete_stmt = delete(models.Message).where(
        models.Message.chat_id == chat_id,
        models.Message.nonce > message.nonce
    )
    db.execute(delete_stmt)
    
    # Создаем новое сообщение
    new_message = models.Message(
        chat_id=chat_id,
        content=message.content,
        sender=message.sender,
        recipient=message.recipient,
        model=message.model.value,
        nonce=message.nonce,
        created_at=message.created_at,
        selected_at=message.selected_at
    )
    
    db.add(new_message)
    db.commit()
    
    # Возвращаем обновленный чат
    return await get_user_chat(db, chat_id, user_id)


async def delete_chat(db: Session, chat_id: int, user_id: int) -> schemas.Chat:
    """
    Удаление чата пользователя (скрытие через флаг visible)
    если чат не найден или не принадлежит пользователю, то ошибка
    """
    # Получаем чат
    chat_stmt = select(models.Chat).where(
        models.Chat.id == chat_id,
        models.Chat.visible == True
    )
    chat = db.execute(chat_stmt).scalar_one_or_none()
    
    # Проверяем, существует ли чат
    if not chat:
        raise exceptions.ChatNotFoundException()
    
    # Проверяем, принадлежит ли чат пользователю
    if chat.user_id != user_id:
        raise exceptions.UserNotChatOwnerException()
    
    # Скрываем чат (устанавливаем visible=False)
    chat.visible = False
    db.add(chat)
    db.commit()
    db.refresh(chat)
    
    # Создаем схему чата для ответа
    return schemas.Chat(
        id=chat.id,
        title=chat.title,
        messages={}  # Пустой словарь сообщений
    )


