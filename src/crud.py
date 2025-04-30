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


async def get_or_create_user(address: str, session: Session) -> models.User:
    # Сначала ищем пользователя по адресу
    stmt = select(models.WalletAddress).where(
        models.WalletAddress.address == address.lower()
    )
    wallet = session.execute(stmt).scalar_one_or_none()
    
    if wallet:
        return wallet.user
    
    # Если адрес не найден, создаем нового пользователя
    user = models.User()
    session.add(user)
    session.commit()
    session.refresh(user)
    
    # Создаем запись адреса для пользователя
    wallet = models.WalletAddress(
        user_id=user.id,
        address=address.lower()
    )
    session.add(wallet)
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


async def get_user_chat(
    db: Session, 
    chat_id: int, 
    user_id: int, 
    from_nonce: Optional[int] = None, 
    to_nonce: Optional[int] = None,
    recipient: Optional[enums.Role] = None
) -> schemas.Chat:
    """
    Получение информации о чате пользователя с сообщениями в диапазоне nonce
    from_nonce - с какого nonce начать (включительно)
    to_nonce - до какого nonce включать (включительно)
    recipient - фильтр по получателю сообщений (если None, то фильтрация не применяется)
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
    
    # Добавляем фильтр по получателю или отправителю, если указан
    if recipient is not None:
        messages_stmt = messages_stmt.where(
            or_(
                models.Message.recipient == recipient,  # Сообщения для пользователя
                models.Message.sender == enums.Role.USER  # Сообщения от пользователя
            )
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
            chat_style=message.chat_style,
            chat_details_level=message.chat_details_level,
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


async def create_chat(db: Session, user_id: int, title: str) -> models.Chat:
    """
    Создание нового чата
    """
    chat = models.Chat(user_id=user_id, title=title)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat

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
        
        chat = await create_chat(db, user_id, title)
        chat_id = chat.id
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
    ).order_by(desc(models.Message.selected_at)).limit(1)
    existing_message = db.execute(existing_message_stmt).scalar_one_or_none()
    
    if existing_message and existing_message.sender == enums.Role.ASSISTANT and message.sender == enums.Role.USER:
        raise exceptions.InvalidNonceException()
    
    # Удаляем все сообщения с большим nonce, но только если регенерируем сообщениие от юзера
    if existing_message and existing_message.sender == enums.Role.USER:
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
        model=message.model,
        nonce=message.nonce,
        chat_style=message.chat_style,
        chat_details_level=message.chat_details_level,
        created_at=message.created_at,
        selected_at=message.selected_at
    )
    
    db.add(new_message)
    db.commit()
    
    # Возвращаем обновленный чат с учетом режима DEBUG
    recipient = None if settings.DEBUG else enums.Role.USER
    return await get_user_chat(db, chat_id, user_id, recipient=recipient)


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


async def update_user_chat_settings(
    user_id: int,
    settings: schemas.UserChatSettings,
    session: Session
) -> models.User:
    """
    Обновление настроек чата пользователя
    """
    user = await get_user_by_id(user_id, session)
    if not user:
        raise exceptions.UserNotFoundException()
    
    # Обновляем все поля, включая None
    user.preferred_chat_model = settings.preferred_chat_model
    user.preferred_chat_style = settings.preferred_chat_style
    user.preferred_chat_details_level = settings.preferred_chat_details_level
    
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


async def update_user_profile(
    user_id: int,
    profile: schemas.UserProfile,
    session: Session
) -> models.User:
    """
    Обновление профиля пользователя
    """
    user = await get_user_by_id(user_id, session)
    if not user:
        raise exceptions.UserNotFoundException()
    
    # Обновляем все поля, включая None
    user.preferred_name = profile.preferred_name
    user.user_context = profile.user_context
    
    session.add(user)
    session.commit()
    session.refresh(user)
    return user

async def add_user_facts(
    user_id: int,
    facts: List[str],
    session: Session
) -> Tuple[models.User, List[str]]:
    """
    Добавление новых фактов о пользователе разом
    """
    user = await get_user_by_id(user_id, session)
    if not user:
        raise exceptions.UserNotFoundException()
    
    # Создаем новые факты
    new_facts = [models.UserFact(
        user_id=user_id,
        description=fact
    ) for fact in facts]

    session.add_all(new_facts)
    session.commit()
    session.refresh(user)
    return user, [fact.description for fact in new_facts]


async def delete_user_facts(
    user_id: int,
    fact_ids: List[int],
    session: Session
) -> Tuple[models.User, List[str]]:
    """
    Удаление фактов о пользователе
    """
    user = await get_user_by_id(user_id, session)
    if not user:
        raise exceptions.UserNotFoundException()

    deleted_facts = []
    # Удаляем факты
    for fact_id in fact_ids:
        # Проверяем, принадлежит ли факт пользователю
        fact_stmt = select(models.UserFact).where(
            models.UserFact.id == fact_id,
            models.UserFact.user_id == user_id
        )
        fact = session.execute(fact_stmt).scalar_one_or_none()
        
        if not fact:
            raise exceptions.FactNotFoundException(f"Fact with id {fact_id} not found")
        
        # Удаляем факт
        session.delete(fact)
        deleted_facts.append(fact.description)

    session.commit()
    session.refresh(user)
    return user, deleted_facts


async def add_user_address(
    user_id: int,
    address: str,
    session: Session
) -> models.WalletAddress:
    """
    Добавление нового адреса кошелька пользователю
    
    Args:
        user_id: ID пользователя
        address: Адрес кошелька (будет преобразован в lowercase)
        session: Сессия базы данных
        
    Returns:
        WalletAddress: Созданный адрес кошелька
        
    Raises:
        exceptions.AddressAlreadyExistsException: Если адрес уже существует в системе
    """
    # Проверяем, существует ли адрес в системе
    stmt = select(models.WalletAddress).where(
        models.WalletAddress.address == address.lower()
    )
    existing_address = session.execute(stmt).scalar_one_or_none()
    
    if existing_address:
        raise exceptions.AddressAlreadyExistsException()
    
    # Создаем новый адрес
    wallet = models.WalletAddress(
        user_id=user_id,
        address=address.lower()
    )
    session.add(wallet)
    session.commit()
    session.refresh(wallet)
    
    return wallet

async def delete_user_address(
    user_id: int,
    address: str,
    session: Session
) -> None:
    """
    Удаление адреса кошелька пользователя
    
    Args:
        user_id: ID пользователя
        address: Адрес кошелька (будет преобразован в lowercase)
        session: Сессия базы данных
        
    Raises:
        exceptions.AddressNotFoundException: Если адрес не найден
        exceptions.LastAddressException: Если это последний адрес пользователя
    """
    # Получаем адрес
    stmt = select(models.WalletAddress).where(
        models.WalletAddress.address == address.lower(),
        models.WalletAddress.user_id == user_id
    )
    wallet = session.execute(stmt).scalar_one_or_none()
    
    if not wallet:
        raise exceptions.AddressNotFoundException()
    
    # Проверяем, не последний ли это адрес пользователя
    stmt = select(func.count()).where(
        models.WalletAddress.user_id == user_id
    )
    address_count = session.execute(stmt).scalar_one()
    
    if address_count <= 1:
        raise exceptions.LastAddressException()
    
    # Удаляем адрес
    session.delete(wallet)
    session.commit()


async def delete_user(user_id: int, session: Session) -> None:
    """
    Полностью удаляет пользователя и все связанные с ним данные:
    - Чаты и сообщения
    - Факты о пользователе
    - Адреса кошельков
    """
    # Получаем пользователя
    user = await get_user_by_id(user_id, session)
    if not user:
        raise exceptions.UserNotFoundException()
    
    # Удаляем все чаты пользователя (это автоматически удалит все сообщения)
    for chat in user.chats:
        session.delete(chat)
    
    # Удаляем все факты пользователя
    for fact in user.facts:
        session.delete(fact)
    
    # Удаляем все адреса кошельков
    for address in user.wallet_addresses:
        session.delete(address)
    
    # Удаляем самого пользователя
    session.delete(user)
    session.commit()

async def get_user_messages_to_analyze(user_id: int, session: Session) -> List[models.Message]:
    """
    Получение сообщений пользователя, которые не были проанализированы.
    Сложный запрос включает в себя получение всех сообщений во всех видимых чатах пользователя, nonce которых больше чем last_analysed_nonce этого чата.
    """
    stmt = select(models.Message).where(
        models.Message.chat_id.in_(
            select(models.Chat.id).where(
                models.Chat.user_id == user_id,
                models.Chat.visible == True
            )
        ),
        models.Message.nonce > models.Chat.last_analysed_nonce,
        models.Message.sender == enums.Role.USER
    )
    return session.execute(stmt).scalars().all()


