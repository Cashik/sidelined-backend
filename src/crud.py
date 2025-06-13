from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict
from sqlalchemy import select, delete, and_, or_, func, desc, update, literal
from sqlalchemy.orm import Session, joinedload

from src import models, schemas, enums, exceptions, utils_base
from src.config.settings import settings
import src.config.subscription_plans as subscriptions

import logging
import time

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
    Полностью удаляет пользователя и все связанные с ним данные.
    Благодаря каскадному удалению на уровне базы данных, все связанные записи 
    (чаты, сообщения, факты, адреса кошельков, промо-коды, шаблоны постов, выбранные проекты)
    будут удалены автоматически.
    """
    if not user:
        raise exceptions.UserNotFoundError()
    
    # Удаляем пользователя - все связанные данные удалятся автоматически благодаря CASCADE
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

async def change_user_credits(db, user_id: int, amount: int):
    """
    Атомарно списывает (или добавляет) кредиты пользователю.
    Если amount < 0 — списание, если > 0 — пополнение.
    """
    result = db.execute(
        update(models.User)
        .where(models.User.id == user_id)
        .values(used_credits_today=models.User.used_credits_today + amount)
    )
    db.commit()
    if result.rowcount == 0:
        raise exceptions.BusinessError(code="change_credits_failed", message="Error use credits")

async def refresh_user_credits(db: Session, user: models.User):
    """
    Сбрасываем использованные кредиты пользователя если прошло больше 24 часов с последнего сброса
    """
    if user.credits_last_update is None or user.credits_last_update < utils_base.now_timestamp() - 60*60*24:
        db.execute(
            update(models.User)
            .where(models.User.id == user.id)
            .values(used_credits_today=0, credits_last_update=utils_base.now_timestamp())
        )
        db.commit()


async def activate_promo_code(db: Session, user: models.User, code: str):
    # Активирует промо-код для пользователя
    # Выкидываем если что-то пошло не так
    
    # Проверяем, существует ли промо-код
    promo_code_stmt = select(models.PromoCode).where(
        models.PromoCode.code == code
    )
    promo_code = db.execute(promo_code_stmt).scalar_one_or_none()
    
    if not promo_code:
        raise exceptions.PromoCodeActivationError("This promo code is not valid")
    
    # Проверяем, истек ли срок действия промо-кода
    if promo_code.valid_until < utils_base.now_timestamp():
        error_msg = "This promo code is expired"
        if settings.DEBUG:
            error_msg += f" (valid_until: {promo_code.valid_until}, now: {utils_base.now_timestamp()})"
        raise exceptions.PromoCodeActivationError(error_msg)
    
    # Проверяем, активировал ли пользователь этот код ранее
    usage_stmt = select(models.PromoCodeUsage).where(
        models.PromoCodeUsage.promo_code_id == promo_code.id,
        models.PromoCodeUsage.user_id == user.id
    )
    usage = db.execute(usage_stmt).scalar_one_or_none()
    
    if usage:
        raise exceptions.PromoCodeActivationError("You have already used this promo code")
    
    # Запоминаем использование промо-кода
    usage = models.PromoCodeUsage(
        promo_code_id=promo_code.id,
        user_id=user.id
    )
    db.add(usage)
    # Активируем промо-код
    user.pro_plan_promo_activated = True
    db.add(user)
    db.commit()
    
    
def create_promo_code(session: Session, code: str, valid_until: int):
    promo_code = models.PromoCode(
        code=code,
        valid_until=valid_until
    )
    session.add(promo_code)
    session.commit()
    
def change_promo_code(session: Session, code: str, valid_until: int):
    promo_code = session.execute(select(models.PromoCode).where(models.PromoCode.code == code)).scalar_one_or_none()
    if not promo_code:
        raise exceptions.PromoCodeNotFoundError()
    promo_code.valid_until = valid_until
    session.add(promo_code)
    session.commit()
    

# projects - проекты, источники, связи между ними и посты

async def get_social_post_by_id(post_id: str, db: Session) -> Optional[models.SocialPost]:
    """
    Получение поста по его ID в социальной сети
    """
    stmt = select(models.SocialPost).where(models.SocialPost.social_id == post_id)
    return db.execute(stmt).scalar_one_or_none()

async def get_social_account_by_id(account_id: str, db: Session) -> Optional[models.SocialAccount]:
    """
    Получение аккаунта по его ID в социальной сети
    """
    stmt = select(models.SocialAccount).where(models.SocialAccount.social_id == account_id)
    return db.execute(stmt).scalar_one_or_none()

async def create_social_account(account: models.SocialAccount, db: Session) -> models.SocialAccount:
    """
    Создание нового аккаунта в социальной сети
    """
    db.add(account)
    db.commit()
    db.refresh(account)
    return account

async def create_social_post(post: models.SocialPost, db: Session) -> models.SocialPost:
    """
    Создание нового поста в социальной сети
    """
    db.add(post)
    db.commit()
    db.refresh(post)
    return post

async def create_or_deny_project_post_mention(project_id: int, post_id: int, db: Session) -> None:
    """
    Создание связи между проектом и постом, если она еще не существует
    """
    # Проверяем, существует ли уже такая связь
    stmt = select(models.ProjectMention).where(
        models.ProjectMention.project_id == project_id,
        models.ProjectMention.post_id == post_id
    )
    existing_mention = db.execute(stmt).scalar_one_or_none()
    
    if not existing_mention:
        # Создаем новую связь
        mention = models.ProjectMention(
            project_id=project_id,
            post_id=post_id
        )
        db.add(mention)
        db.commit()

async def create_social_post_statistic(statistic: models.SocialPostStatistic, db: Session) -> models.SocialPostStatistic:
    """
    Создание статистики для поста
    """
    db.add(statistic)
    db.commit()
    db.refresh(statistic)
    return statistic

async def get_posts(
    filter: schemas.FeedFilter,
    sort_type: enums.SortType,
    db: Session,
    limit: int = 100,
) -> List[models.SocialPost]:
    """
    Возвращает отфильтрованную и отсортированную ленту постов.

    Параметры
    ---------
    filter : schemas.FeedFilter
        Условия выборки.
    sort_type : enums.SortType
        Тип сортировки (NEW | POPULAR).
    db : Session
        SQLAlchemy-сессия.
    limit : int
        Максимальное число постов (по ТЗ = 100).
    """

    # Если оба флага выключены – нечего отдавать
    if not filter.include_project_sources and not filter.include_other_sources:
        return []

    day_ago = utils_base.now_timestamp() - 60 * 60 * 24

    # базовый SELECT
    query = (
        select(models.SocialPost)
        .options(
            joinedload(models.SocialPost.statistic),
            joinedload(models.SocialPost.account).joinedload(models.SocialAccount.projects),
        )
        .where(models.SocialPost.posted_at >= day_ago)
    )

    # --- фильтр по project_ids через EXISTS, чтобы избежать дубликатов ---
    if filter.projects_ids:
        mention_exists = (
            select(literal(1))
            .select_from(models.ProjectMention)
            .where(
                and_(
                    models.ProjectMention.post_id == models.SocialPost.id,
                    models.ProjectMention.project_id.in_(filter.projects_ids),
                )
            )
            .exists()
        )
        query = query.where(mention_exists)

    # --- фильтр по типу источников (привязан / не привязан к проекту) ---
    if filter.include_project_sources != filter.include_other_sources:
        account_link_exists = (
            select(literal(1))
            .select_from(models.ProjectAccountStatus)
            .where(models.ProjectAccountStatus.account_id == models.SocialPost.account_id)
            .exists()
        )

        if filter.include_project_sources:
            query = query.where(account_link_exists)
        else:
            query = query.where(~account_link_exists)

    # --- фильтр по engagement score ---
    if filter.include_other_sources:
        # Коррелированный подзапрос: берём только одну – самую свежую – статистику для поста
        latest_score_subq = (
            select(
                (
                    func.coalesce(models.SocialPostStatistic.views, 0) * 0.001
                    + func.coalesce(models.SocialPostStatistic.likes, 0) * 1
                    + func.coalesce(models.SocialPostStatistic.reposts, 0) * 3
                    + func.coalesce(models.SocialPostStatistic.comments, 0) * 4
                )
            )
            .where(models.SocialPostStatistic.post_id == models.SocialPost.id)
            .order_by(desc(models.SocialPostStatistic.created_at))
            .limit(1)
            .scalar_subquery()
        )
        query = query.where(latest_score_subq >= settings.POST_FEED_MINIMAL_ENGAGEMENT_SCORES)

    # --- сортировка ---
    if sort_type == enums.SortType.NEW:
        query = query.order_by(desc(models.SocialPost.posted_at))
    elif sort_type == enums.SortType.POPULAR:
        # Коррелированный подзапрос: берём только одну – самую свежую – статистику для поста
        latest_score_subq = (
            select(
                (
                    func.coalesce(models.SocialPostStatistic.views, 0) * 0.001
                    + func.coalesce(models.SocialPostStatistic.likes, 0) * 1
                    + func.coalesce(models.SocialPostStatistic.reposts, 0) * 2
                    + func.coalesce(models.SocialPostStatistic.comments, 0) * 4
                )
            )
            .where(models.SocialPostStatistic.post_id == models.SocialPost.id)
            .order_by(desc(models.SocialPostStatistic.created_at))
            .limit(1)
            .scalar_subquery()
        )

        popularity_expr = func.coalesce(latest_score_subq, 0)
        query = query.order_by(desc(popularity_expr))
    else:
        # На случай добавления новых типов сортировки в будущем
        raise exceptions.BusinessError(code="invalid_sort_type", message="Invalid sort type")

    # --- лимит ---
    query = query.limit(limit)

    # выполняем
    result = db.execute(query).unique().scalars().all()
    return result

async def get_projects_all(db: Session) -> List[models.Project]:
    stmt = select(models.Project)
    return db.execute(stmt).scalars().all()

async def get_projects_selected_by_user(user: models.User, db: Session) -> List[models.Project]:
    stmt = (
        select(models.Project)
        .join(models.UserSelectedProject)
        .where(models.UserSelectedProject.user_id == user.id)
    )
    return db.execute(stmt).scalars().all()

async def select_projects(request: schemas.SelectProjectsRequest, user: models.User, db: Session) -> None:
    """Обновляет список выбранных проектов для пользователя"""
    # Удаляем все существующие выбранные проекты пользователя
    stmt = delete(models.UserSelectedProject).where(
        models.UserSelectedProject.user_id == user.id
    )
    db.execute(stmt)
    
    # Добавляем новые выбранные проекты
    selected_projects = [
        models.UserSelectedProject(
            user_id=user.id,
            project_id=project_id
        )
        for project_id in request.project_ids
    ]
    
    db.add_all(selected_projects)
    db.commit()
    

def create_or_update_project_by_name(session: Session, project: models.Project) -> models.Project:
    """
    Создает или обновляет проект по его имени.
    Если проект с таким именем уже существует, обновляет его данные.
    """
    # Ищем проект по имени
    stmt = select(models.Project).where(models.Project.name == project.name)
    existing_project = session.execute(stmt).scalar_one_or_none()
    
    if existing_project:
        # Обновляем существующий проект
        existing_project.description = project.description
        existing_project.url = project.url
        existing_project.keywords = project.keywords
        session.add(existing_project)
        session.commit()
        session.refresh(existing_project)
        return existing_project
    else:
        # Создаем новый проект
        session.add(project)
        session.commit()
        session.refresh(project)
        return project

def create_or_update_social_media_by_social_name(session: Session, account: models.SocialAccount) -> models.SocialAccount:
    """
    Создает или обновляет аккаунт социальной сети по его social_login.
    Если аккаунт с таким social_login уже существует, обновляет его данные.
    """
    # Ищем аккаунт по social_login
    stmt = select(models.SocialAccount).where(models.SocialAccount.social_login == account.social_login)
    existing_account = session.execute(stmt).scalar_one_or_none()
    
    if existing_account:
        # Обновляем существующий аккаунт
        existing_account.name = account.name
        existing_account.social_id = account.social_id
        session.add(existing_account)
        session.commit()
        session.refresh(existing_account)
        return existing_account
    else:
        # Создаем новый аккаунт
        session.add(account)
        session.commit()
        session.refresh(account)
        return account

def create_or_update_project_account_status(
    session: Session,
    project_id: int,
    account_id: int,
    status_type: enums.ProjectAccountStatusType
) -> models.ProjectAccountStatus:
    """
    Создает или обновляет связь между проектом и аккаунтом.
    Если связь уже существует, обновляет тип связи.
    """
    # Ищем существующую связь
    stmt = select(models.ProjectAccountStatus).where(
        models.ProjectAccountStatus.project_id == project_id,
        models.ProjectAccountStatus.account_id == account_id
    )
    existing_status = session.execute(stmt).scalar_one_or_none()
    
    if existing_status:
        # Обновляем тип связи
        existing_status.type = status_type
        session.add(existing_status)
        session.commit()
        session.refresh(existing_status)
        return existing_status
    else:
        # Создаем новую связь
        status = models.ProjectAccountStatus(
            project_id=project_id,
            account_id=account_id,
            type=status_type
        )
        session.add(status)
        session.commit()
        session.refresh(status)
        return status

async def delete_old_posts(db: Session, older_than_timestamp: int) -> int:
    """
    Удаляет все посты старше указанного timestamp.
    
    Args:
        db: Сессия базы данных
        older_than_timestamp: Timestamp, посты старше которого будут удалены
        
    Returns:
        Количество удаленных постов
    """
    # Сначала удаляем связанные данные (статистику и упоминания)
    # Получаем ID постов для удаления
    posts_to_delete_stmt = select(models.SocialPost.id).where(
        models.SocialPost.posted_at < older_than_timestamp
    )
    post_ids = db.execute(posts_to_delete_stmt).scalars().all()
    
    if not post_ids:
        return 0
    
    # Удаляем статистику постов
    delete_stats_stmt = delete(models.SocialPostStatistic).where(
        models.SocialPostStatistic.post_id.in_(post_ids)
    )
    db.execute(delete_stats_stmt)
    
    # Удаляем упоминания проектов в постах
    delete_mentions_stmt = delete(models.ProjectMention).where(
        models.ProjectMention.post_id.in_(post_ids)
    )
    db.execute(delete_mentions_stmt)
    
    # Удаляем сами посты
    delete_posts_stmt = delete(models.SocialPost).where(
        models.SocialPost.posted_at < older_than_timestamp
    )
    result = db.execute(delete_posts_stmt)
    
    db.commit()
    
    return result.rowcount

async def get_brain_settings(user: models.User, session: Session) -> schemas.PersonalizationSettings:
    """
    Получение настроек персонализации пользователя
    """
    if not user:
        raise exceptions.UserNotFoundError()
    
    if user.personalization_brain_settings:
        try:
            # Используем безопасную схему для извлечения из базы данных
            brain_settings_safe = schemas.PersonalizationSettingsSafe.model_validate(user.personalization_brain_settings)
            logger.info(f"Retrieved brain settings for user {user.id}: {brain_settings_safe}")
            
            # Преобразуем в обычную схему для ответа (простое копирование полей)
            return schemas.PersonalizationSettings.model_validate(brain_settings_safe.model_dump())
        except Exception as e:
            logger.warning(f"Error validating brain settings for user {user.id}: {e}")
            # Возвращаем настройки по умолчанию при ошибке валидации
            return schemas.PersonalizationSettings(
                user_social_login="",
                style=schemas.StyleSettings(),
                content=schemas.ContentSettings()
            )
    else:
        # Возвращаем настройки по умолчанию, если их нет в базе
        return schemas.PersonalizationSettings(
            user_social_login="",
            style=schemas.StyleSettings(),
            content=schemas.ContentSettings()
        )


async def set_brain_settings(
    request: schemas.PersonalizationSettings,
    user: models.User,
    session: Session
) -> schemas.PersonalizationSettings:
    """
    Установка настроек персонализации пользователя
    """
    if not user:
        raise exceptions.UserNotFoundError()
    
    # Сохраняем настройки в базу данных
    user.personalization_brain_settings = request.model_dump(mode="json")
    logger.info(f"Updated brain settings for user {user.id}: {user.personalization_brain_settings}")
    session.add(user)
    session.commit()
    session.refresh(user)
    
    # Возвращаем сохраненные настройки
    return await get_brain_settings(user, session)


async def get_feed_templates(session: Session, projects_ids: Optional[List[int]] = None, limit: int = 50) -> List[schemas.PostExample]:
    """
    Получение шаблонов для auto-yaps по выбранным проектам.
    Если projects_ids не указаны, возвращает шаблоны для всех проектов.
    """
    # Получаем шаблоны для выбранных проектов или для всех проектов
    stmt = select(models.PostTemplate)
    
    # Если указаны конкретные проекты, фильтруем по ним
    if projects_ids:
        stmt = stmt.where(models.PostTemplate.project_id.in_(projects_ids))
    
    # Добавляем сортировку и лимит
    stmt = stmt.order_by(desc(models.PostTemplate.created_at)).limit(limit)
    
    templates = session.execute(stmt).scalars().all()
    
    return [schemas.PostExample.model_validate(template) for template in templates]


async def get_project_feed(project: models.Project, period_timestamp: int, session: Session) -> List[models.SocialPost]:
    """
    Получение ленты постов для выбранного проекта за указанный период
    
    Args:
        project: Проект, для которого нужно получить посты
        period_timestamp: Период в секундах (например, 86400 для последних 24 часов)
        session: Сессия базы данных
        
    Returns:
        Список постов, связанных с проектом за указанный период
    """
    latest_fresh_post_timestamp = utils_base.now_timestamp() - period_timestamp
    
    # Получаем все посты для выбранного проекта, учитывая, что проект и пост связаны через project_mention
    stmt = (
        select(models.SocialPost)
        .options(
            joinedload(models.SocialPost.statistic),
            joinedload(models.SocialPost.account),
        )
        .join(models.ProjectMention, models.ProjectMention.post_id == models.SocialPost.id)
        .where(
            models.ProjectMention.project_id == project.id,
            models.SocialPost.posted_at >= latest_fresh_post_timestamp
        )
        .order_by(desc(models.SocialPost.posted_at))
    )
    
    return session.execute(stmt).unique().scalars().all()


async def create_post_examples(templates: List[schemas.PostExampleCreate], db: Session) -> List[schemas.PostExample]:
    """
    Создание шаблонных постов для юзера
    """
    # Создаем объекты модели
    post_examples_db = []
    for template in templates:
        post_example_db = models.PostTemplate(
            project_id=template.project_id,
            post_text=template.post_text
        )
        db.add(post_example_db)
        post_examples_db.append(post_example_db)
    
    # Сохраняем все объекты в базу данных
    db.commit()
    
    # Обновляем объекты, чтобы получить сгенерированные поля (id, created_at)
    for post_example_db in post_examples_db:
        db.refresh(post_example_db)
    
    # Преобразуем в схемы
    result: List[schemas.PostExample] = []
    for post_example_db in post_examples_db:
        result.append(schemas.PostExample.model_validate(post_example_db))
    
    return result

async def get_project_feed_posts(
    db: Session,
    project_ids: List[int],
    include_project_sources: bool = True,
    sort_type: enums.SortType = enums.SortType.POPULAR,
    limit: int = 100,
) -> List[models.SocialPost]:
    """
    Возвращает посты, связанные с проектом через ProjectMention за последний день.
    Если include_project_sources == False, исключает посты от связанных аккаунтов.
    """
    day_ago = utils_base.now_timestamp() - 60 * 60 * 24
    # Базовый запрос: посты, связанные с проектом через ProjectMention
    query = (
        select(models.SocialPost)
        .options(
            joinedload(models.SocialPost.statistic),
            joinedload(models.SocialPost.account).joinedload(models.SocialAccount.projects),
        )
        .join(models.ProjectMention, models.ProjectMention.post_id == models.SocialPost.id)
        .where(
            models.ProjectMention.project_id.in_(project_ids),
            models.SocialPost.posted_at >= day_ago
        )
    )
    if not include_project_sources:
        # Получаем id аккаунтов, связанных с проектом
        account_ids_subq = (
            select(models.ProjectAccountStatus.account_id)
            .where(models.ProjectAccountStatus.project_id.in_(project_ids))
        )
        query = query.where(~models.SocialPost.account_id.in_(account_ids_subq))
    # Сортировка
    if sort_type == enums.SortType.NEW:
        query = query.order_by(desc(models.SocialPost.posted_at))
    elif sort_type == enums.SortType.POPULAR:
        latest_score_subq = (
            select(
                (
                    func.coalesce(models.SocialPostStatistic.views, 0) * 0.001
                    + func.coalesce(models.SocialPostStatistic.likes, 0) * 1
                    + func.coalesce(models.SocialPostStatistic.reposts, 0) * 2
                    + func.coalesce(models.SocialPostStatistic.comments, 0) * 4
                )
            )
            .where(models.SocialPostStatistic.post_id == models.SocialPost.id)
            .order_by(desc(models.SocialPostStatistic.created_at))
            .limit(1)
            .scalar_subquery()
        )
        popularity_expr = func.coalesce(latest_score_subq, 0)
        query = query.order_by(desc(popularity_expr))
    query = query.limit(limit)
    result = db.execute(query).unique().scalars().all()
    return result

async def get_project_news_posts(
    db: Session,
    project_ids: List[int],
    sort_type: enums.SortType = enums.SortType.POPULAR,
    limit: int = 100,
) -> List[models.SocialPost]:
    """
    Возвращает все посты всех аккаунтов, связанных с проектом, за последний день.
    """
    day_ago = utils_base.now_timestamp() - 60 * 60 * 24
    # Получаем id аккаунтов, связанных с проектом
    account_ids_subq = (
        select(models.ProjectAccountStatus.account_id)
        .where(models.ProjectAccountStatus.project_id.in_(project_ids))
    )
    query = (
        select(models.SocialPost)
        .options(
            joinedload(models.SocialPost.statistic),
            joinedload(models.SocialPost.account).joinedload(models.SocialAccount.projects),
        )
        .where(
            models.SocialPost.account_id.in_(account_ids_subq),
            models.SocialPost.posted_at >= day_ago
        )
    )
    # Сортировка
    if sort_type == enums.SortType.NEW:
        query = query.order_by(desc(models.SocialPost.posted_at))
    elif sort_type == enums.SortType.POPULAR:
        latest_score_subq = (
            select(
                (
                    func.coalesce(models.SocialPostStatistic.views, 0) * 0.001
                    + func.coalesce(models.SocialPostStatistic.likes, 0) * 1
                    + func.coalesce(models.SocialPostStatistic.reposts, 0) * 2
                    + func.coalesce(models.SocialPostStatistic.comments, 0) * 4
                )
            )
            .where(models.SocialPostStatistic.post_id == models.SocialPost.id)
            .order_by(desc(models.SocialPostStatistic.created_at))
            .limit(1)
            .scalar_subquery()
        )
        popularity_expr = func.coalesce(latest_score_subq, 0)
        query = query.order_by(desc(popularity_expr))
    query = query.limit(limit)
    result = db.execute(query).unique().scalars().all()
    return result


async def get_account_payouts(db: Session, project_id: int, account_id: int) -> List[models.ScorePayout]:
    """
    Получение всех payouts для аккаунта в проекте.
    """
    return db.query(models.ScorePayout).filter(
        models.ScorePayout.project_id == project_id,
        models.ScorePayout.social_account_id == account_id
    ).all()
    
async def get_project_leaderboard_last_ts(db: Session, project_id: int) -> int:
    """
    Получение timestamp последнего обновления leaderboard для проекта.
    """
    return db.query(models.ProjectLeaderboardHistory).filter(
        models.ProjectLeaderboardHistory.project_id == project_id
    ).order_by(desc(models.ProjectLeaderboardHistory.created_at)).first().created_at