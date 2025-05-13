from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict
from sqlalchemy import select, delete, or_, and_, func, desc, update
from sqlalchemy.orm import Session

from src import models, schemas, enums, exceptions, utils_base
from src.config.settings import settings

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
        # Создаем объект ChatSummary
        result.append(schemas.ChatSummary(
            id=chat.id,
            title=chat.title,
            created_at=chat.created_at
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
        raise exceptions.ChatNotFoundError()
    
    # Проверяем, принадлежит ли чат пользователю
    if chat.user_id != user_id:
        raise exceptions.UserNotChatOwnerError()
    
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
    messages_dict: Dict[int, List[schemas.MessageUnion]] = {}
    for message in messages:
        # Преобразуем модель в словарь
        schema_message = schemas.MessageUnionAdapter.validate_python(message)
        
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
    return schemas.Chat(
        id=chat.id,
        title=chat.title,
        messages={}
    )

async def get_next_nonce(db: Session, chat_id: int, user_id: int) -> int:
    """
    Получение следующего доступного nonce для чата
    """
    stmt = select(func.max(models.Message.nonce)).where(
        models.Message.chat_id == chat_id,
        models.Message.sender == enums.Role.USER
    )
    max_nonce = db.execute(stmt).scalar_one_or_none() or 0
    return max_nonce + 1

async def add_chat_messages(db: Session, chat_id: int, messages: List[schemas.MessageUnion], user_id: int) -> schemas.Chat:
    """
    Добавление нескольких сообщений в чат
    """
    def shema_message_to_model(message: schemas.MessageUnion) -> models.Message:
        return models.Message(
            type=message.type,
            chat_id=chat_id,
            content=message.content.model_dump(mode="json"),
            sender=message.sender,
            recipient=message.recipient,
            nonce=message.nonce,
            generation_time_ms=message.generation_time_ms
        )
        
    try:
        for message in messages:
            db.add(shema_message_to_model(message))
        
        db.commit()
    except Exception as e:
        raise exceptions.AddMessageError()
    
    return await get_user_chat(db, chat_id, user_id)
    

async def delete_chat_messages(db: Session, chat_id: int, from_nonce: int) -> schemas.Chat:
    """
    Удаление сообщений в чате после определенного nonce
    """
    stmt = delete(models.Message).where(
        models.Message.chat_id == chat_id,
        models.Message.nonce >= from_nonce,
    )
    db.execute(stmt)
    db.commit()



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
        raise exceptions.ChatNotFoundError()
    
    # Проверяем, принадлежит ли чат пользователю
    if chat.user_id != user_id:
        raise exceptions.UserNotChatOwnerError()
    
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
    settings: schemas.MessageGenerationSettings,
    session: Session
) -> models.User:
    """
    Обновление настроек чата пользователя
    """
    user = await get_user_by_id(user_id, session)
    if not user:
        raise exceptions.UserNotFoundError()
    
    # Обновляем все поля, включая None
    user.chat_settings = settings.model_dump(mode="json")
    logger.info(f"Updated chat settings: {user.chat_settings}")
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


async def update_user_profile(
    user: models.User,
    profile: schemas.UserProfile,
    session: Session
) -> models.User:
    """
    Обновление профиля пользователя
    """
    if not user:
        raise exceptions.UserNotFoundError()
    
    # Обновляем все поля, включая None
    user.preferred_name = profile.preferred_name
    user.user_context = profile.user_context
    
    session.add(user)
    session.commit()
    session.refresh(user)
    return user

async def add_user_facts(
    user: models.User,
    facts: List[str],
    session: Session
) -> Tuple[models.User, List[str]]:
    """
    Добавление новых фактов о пользователе разом
    """
    if not user:
        raise exceptions.UserNotFoundError()
    
    # Создаем новые факты
    new_facts = [models.UserFact(
        user_id=user.id,
        description=fact
    ) for fact in facts]

    session.add_all(new_facts)
    session.commit()
    session.refresh(user)
    return user, [fact.description for fact in new_facts]


async def delete_user_facts(
    user: models.User,
    fact_ids: List[int],
    session: Session
) -> Tuple[models.User, List[str]]:
    """
    Удаление фактов о пользователе
    """
    if not user:
        raise exceptions.UserNotFoundError()

    deleted_facts = []
    # Удаляем факты
    for fact_id in fact_ids:
        # Проверяем, принадлежит ли факт пользователю
        fact_stmt = select(models.UserFact).where(
            models.UserFact.id == fact_id,
            models.UserFact.user_id == user.id
        )
        fact = session.execute(fact_stmt).scalar_one_or_none()
        
        if not fact:
            raise exceptions.FactNotFoundError(f"Fact with id {fact_id} not found")
        
        # Удаляем факт
        session.delete(fact)
        deleted_facts.append(fact.description)

    session.commit()
    session.refresh(user)
    return user, deleted_facts


async def add_user_address(
    user: models.User,
    address: str,
    session: Session
) -> models.WalletAddress:
    # Проверяем, существует ли адрес в системе
    stmt = select(models.WalletAddress).where(
        models.WalletAddress.address == address.lower()
    )
    existing_address = session.execute(stmt).scalar_one_or_none()
    
    if existing_address:
        raise exceptions.AddressAlreadyExistsError()
    
    # Создаем новый адрес
    wallet = models.WalletAddress(
        user_id=user.id,
        address=address.lower()
    )
    session.add(wallet)
    session.commit()
    session.refresh(wallet)
    
    return wallet

async def delete_user_address(
    user: models.User,
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
        exceptions.AddressNotFoundError: Если адрес не найден
        exceptions.LastAddressError: Если это последний адрес пользователя
    """
    # Получаем адрес
    stmt = select(models.WalletAddress).where(
        models.WalletAddress.address == address.lower(),
        models.WalletAddress.user_id == user.id
    )
    wallet = session.execute(stmt).scalar_one_or_none()
    
    if not wallet:
        raise exceptions.AddressNotFoundError()
    
    # Проверяем, не последний ли это адрес пользователя
    stmt = select(func.count()).where(
        models.WalletAddress.user_id == user.id
    )
    address_count = session.execute(stmt).scalar_one()
    
    if address_count <= 1:
        raise exceptions.LastAddressError()
    
    # Удаляем адрес
    session.delete(wallet)
    session.commit()


async def delete_user(user: models.User, session: Session) -> None:
    """
    Полностью удаляет пользователя и все связанные с ним данные:
    - Чаты и сообщения
    - Факты о пользователе
    - Адреса кошельков
    """
    if not user:
        raise exceptions.UserNotFoundError()
    
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

async def get_user_messages_to_analyze(user: models.User, session: Session) -> List[models.Message]:
    """
    Получение сообщений пользователя, которые не были проанализированы.
    Для каждого чата пользователя возвращает только те сообщения, у которых nonce > last_analysed_nonce.
    """
    stmt = (
        select(models.Message)
        .join(models.Chat, models.Message.chat_id == models.Chat.id)
        .where(
            models.Chat.user_id == user.id,
            models.Chat.visible == True,
            models.Message.nonce > models.Chat.last_analysed_nonce,
            models.Message.sender == enums.Role.USER,
            models.Message.type == enums.MessageType.TEXT
        )
    )
    return session.execute(stmt).scalars().all()

async def change_user_credits(db, user: models.User, amount: int):
    """
    Атомарно списывает (или добавляет) кредиты пользователю.
    Если amount < 0 — списание, если > 0 — пополнение.
    """
    result = db.execute(
        update(models.User)
        .where(models.User.id == user.id)
        .values(credits=models.User.credits + amount)
    )
    db.commit()
    if result.rowcount == 0:
        raise exceptions.BusinessError(code="change_credits_failed", message="Не удалось изменить баланс кредитов (возможно, недостаточно средств)")

async def refresh_user_credits(db, user: models.User):
    """
    Обновляет баланс кредитов пользователя если прошло больше 24 часов с последнего обновления
    """
    if user.credits_last_update is None or user.credits_last_update < utils_base.now_timestamp() - 60*60*24:
        db.execute(
            update(models.User)
            .where(models.User.id == user.id)
            .values(credits=settings.DEFAULT_CREDITS, credits_last_update=utils_base.now_timestamp())
        )
        db.commit()



