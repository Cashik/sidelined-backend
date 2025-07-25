from enum import Enum
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Request
from fastapi.responses import StreamingResponse, RedirectResponse
from typing import AsyncGenerator, Dict, Any, List, Optional, Tuple, Union
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, joinedload, selectinload
import logging
import json
import jsonschema
from jsonschema import validate, ValidationError
import time
from uuid import uuid4

from src import schemas, enums, models, crud, utils, utils_base, exceptions
from src.core.middleware import get_current_user
from src.database import get_session
from src.services import user_context_service
from src.config.settings import settings
from src.config.mcp_servers import get_toolboxes, mcp_servers
from src.services.prompt_service import PromptService
from src.config import ai_models, subscription_plans
from src.services.x_api_service import XApiService
from src.services.x_oauth_service import XOAuthService
from src.config.subscription_plans import get_subscription_plan
from src.services.redis_client import redis_client
from src.services.cache_service import LeaderboardPeriod

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter(prefix="/yaps", tags=["Auto-Yaps"])

class Project(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    url: Optional[str] = None
    icon_url: str = ""

class GetProjectsResponse(BaseModel):
    projects: List[Project]


@router.get("/projects/all", response_model=GetProjectsResponse)
async def get_projects(db: Session = Depends(get_session)):
    """
    Получение списка всех отслеживаемых проектов приложения
    """
    projects_db: List[models.Project] = await crud.get_projects_all(db)
    projects = [Project(id=project.id, name=project.name, description=project.description, url=project.url, icon_url=project.icon_url) for project in projects_db]
    return GetProjectsResponse(projects=projects)


@router.get("/projects/selected", response_model=GetProjectsResponse)
async def get_selected_projects(user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    """
    Получение списка отслеживаемых проектов для конкретного пользователя
    """
    projects_db: List[models.Project] = await crud.get_projects_selected_by_user(user, db)
    projects = [Project(id=project.id, name=project.name, description=project.description, url=project.url, icon_url=project.icon_url) for project in projects_db]
    return GetProjectsResponse(projects=projects)


@router.post("/projects/selected")
async def select_projects(request: schemas.SelectProjectsRequest, user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    """
    Выбор проектов для отслеживания
    """
    await crud.select_projects(request, user, db)
    return {"message": "Projects selected successfully"}


@router.get("/feed", response_model=schemas.GetFeedResponse)
async def get_feed(
    projects_ids: Optional[List[int]] = Query(None, description="List of project IDs to filter by"),
    sort_type: enums.SortType = enums.SortType.POPULAR,
    db: Session = Depends(get_session)
):
    """
    Получение ленты постов, связанных с проектом (по упоминаниям), без постов от официальных источников.
    Теперь используется кеш топ-100 постов по engagement за сутки для каждого проекта.
    """
    start_ts = utils_base.now_timestamp()
    if not projects_ids:
        projects = await crud.get_projects_all(db)
    else:
        projects = db.query(models.Project).filter(models.Project.id.in_(projects_ids)).all()
    all_posts = []
    for project in projects:
        posts = await utils.get_feed(project, db)
        all_posts.extend(posts)
    # Сортировка
    all_posts.sort(key=lambda p: p.stats.favorite_count + p.stats.retweet_count*3 + p.stats.reply_count*4 + p.stats.views_count*0.001, reverse=True)
    # обрезаем до 100 постов
    all_posts = all_posts[:100]
    # если необходимо, сортируем по дате
    if sort_type == enums.SortType.NEW:
        all_posts.sort(key=lambda p: p.created_timestamp, reverse=True)
    logger.info(f"[Feed] Found {len(all_posts)} posts in {utils_base.now_timestamp() - start_ts} seconds")
    return schemas.GetFeedResponse(posts=all_posts)

@router.get("/news", response_model=schemas.GetFeedResponse)
async def get_news(
    projects_ids: Optional[List[int]] = Query(None, description="List of project IDs to filter by"),
    sort_type: enums.SortType = enums.SortType.NEW,
    db: Session = Depends(get_session)
):
    """
    Получение всех постов всех аккаунтов, связанных с проектом, за последний день.
    """
    if not projects_ids:
        projects = await crud.get_projects_all(db)
        projects_ids = [p.id for p in projects]
    posts = await crud.get_project_news_posts(
        db=db,
        project_ids=projects_ids,
        sort_type=sort_type,
        limit=100,
    )
    posts_schemas = utils.convert_posts_to_schemas(posts)
    return schemas.GetFeedResponse(posts=posts_schemas)


@router.get("/auto/settings", response_model=schemas.PersonalizationSettings)
async def get_autoyaps_settings(user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    """
    Получение настроек для нейросети
    """
    return await crud.get_brain_settings(user, db)

@router.post("/auto/settings", response_model=schemas.PersonalizationSettings)
async def set_autoyaps_settings(request: schemas.PersonalizationSettings, user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    """
    Установка настроек для нейросети
    """
    return await crud.set_brain_settings(request, user, db)


class FeedTemplatesResponse(BaseModel):
    templates: List[schemas.PostExample]

@router.get("/auto/templates", response_model=FeedTemplatesResponse)
async def get_feed_templates(
    projects_ids: Optional[List[int]] = Query(None, description="List of project IDs to filter by"),
    db: Session = Depends(get_session)
):
    """
    Получение шаблонов для auto-yaps
    """
    return FeedTemplatesResponse(
        templates=await crud.get_feed_templates(db, projects_ids),
    )

# TODO: удалить
@router.post("/auto/templates", response_model=FeedTemplatesResponse)
async def create_feed_template(db: Session = Depends(get_session)):
    """
    Создание шаблонов для auto-yaps
    """
    projects = await crud.get_projects_all(db)
    new_templates = await utils.create_project_autoyaps(projects[0], db)
    return FeedTemplatesResponse(
        templates=new_templates,
    )

class YapsPersonalizationRequest(BaseModel):
    text: str

class YapsPersonalizationResponse(BaseModel):
    text: str
    variants: List[str] = Field(description="Few variants of personalized tweets")
    
@router.post("/personalize", response_model=YapsPersonalizationResponse)
async def personalize(
    request: YapsPersonalizationRequest,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Персонализация текста для авто-постов
    """
    PERSONALIZATION_CREDITS_COST = 1
    # Получаем тариф пользователя по актуальному типу подписки
    user_plan: enums.SubscriptionPlanType = await utils.check_user_subscription(user, db)
    user_subscription = get_subscription_plan(user_plan)
    # Проверяем лимит кредитов
    if user.used_credits_today + PERSONALIZATION_CREDITS_COST > user_subscription.max_credits:
        raise exceptions.APIError(code="out_of_credits", message="Out of credits. Try again tomorrow.", status_code=403)

    # Получаем настройки персонализации пользователя
    personalization_settings = await crud.get_brain_settings(user, db)
    # Генерируем 3 персонализированных варианта
    variants = await utils.generate_personalized_tweets(
        original_text=request.text,
        personalization_settings=personalization_settings,
        count=3
    )
    # Списываем кредиты
    await crud.change_user_credits(db, user.id, PERSONALIZATION_CREDITS_COST)
    return YapsPersonalizationResponse(text=variants[0], variants=variants)

    
class LeaderboardResponse(BaseModel):
    users: List[schemas.LeaderboardUser]
    

@router.get("/leaderboard", response_model=LeaderboardResponse)
async def get_leaderboard(
    period: LeaderboardPeriod = Query(LeaderboardPeriod.ONE_DAY, description="Период для лидерборда: 1d, 1w, 1m, all"),
    project_id: int = Query(..., description="ID проекта для лидерборда (обязательный параметр)"),
    db: Session = Depends(get_session)
):
    """
    Получение лидерборда за определённый период (1d, 1w, 1m, all) для указанного проекта
    """
    # 1. Получаем проект для лидерборда
    project = db.query(models.Project).filter(
        models.Project.id == project_id,
        models.Project.is_leaderboard_project == True
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"Leaderboard project with id '{project_id}' not found")

    users = await utils.get_leaderboard(project, period, db)
    users_schemas = [schemas.LeaderboardUser(**user) for user in users]
    # sort by score
    users_schemas.sort(key=lambda u: u.scores, reverse=True)
    logger.info(f"[Leaderboard] Found {len(users)} users for project {project.name}, period {period}")
    # 3. Собираем leaderboard
    return LeaderboardResponse(users=users_schemas)

# --- X (Twitter) OAuth ---

@router.get("/x/connect-url")
async def get_x_connect_url(user: models.User = Depends(get_current_user)):
    """
    Получить ссылку для подключения X (Twitter) аккаунта через OAuth (с state)
    """
    state = str(uuid4())
    await redis_client.setex(f"oauth_state:{state}", 600, user.id)  # 10 минут TTL
    oauth = XOAuthService()
    url = oauth.get_authorize_url(state)
    return {"url": url}

@router.get("/x/callback")
async def x_oauth_callback(code: str, state: str, db: Session = Depends(get_session)):
    """
    Callback for OAuth X (Twitter). Saves the user's X login to the User model by state.
    On success, redirects to frontend URL from settings.TWITTER_SUCCESS_REDIRECT_URI.
    """
    user_id = await redis_client.get(f"oauth_state:{state}")
    await redis_client.delete(f"oauth_state:{state}")
    if not user_id:
        raise HTTPException(status_code=400, detail="Invalid or expired state")
    user = db.get(models.User, int(user_id))  # db.get is sync
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    oauth = XOAuthService()
    token_data = await oauth.fetch_token(code)
    access_token = token_data.get("access_token")
    if not access_token:
        raise HTTPException(status_code=400, detail="No access_token from X")
    login = await oauth.get_user_login(access_token)
    # Check that this login is not already used
    existing = db.execute(db.query(models.User).filter(models.User.twitter_login == login))
    if existing.scalars().first():
        raise HTTPException(status_code=409, detail="This X account is already linked to another user")
    user.twitter_login = login

    # --- Создание SocialAccount и установка xscore ---
    social_account = db.query(models.SocialAccount).filter(models.SocialAccount.social_login == login).first()
    if not social_account:
        # Получаем rest_id через XApiService (лучше, чем просто login)
        try:
            x_api = XApiService()
            rest_id = await x_api.get_user_social_id(login)
            social_account = models.SocialAccount(
                social_id=rest_id,
                social_login=login,
                name=login
            )
            db.add(social_account)
            db.commit()
            db.refresh(social_account)
        except Exception as e:
            logger.error(f"Do not get rest_id for {login}: {e}")
            rest_id = login  # fallback
        
    # Устанавливаем xscore, если не установлен
    if social_account and social_account.twitter_scout_score is None:
        try:
            from src import utils
            xscore = utils.get_social_account_xscore(social_account)
            social_account.twitter_scout_score = xscore
            social_account.twitter_scout_score_updated_at = int(time.time())
            db.commit()
        except Exception as e:
            logger.error(f"Cannot get xscore for {login}: {e}")
    # --- конец блока ---

    db.commit()
    # Redirect to frontend with success params
    redirect_url = f"{settings.TWITTER_SUCCESS_REDIRECT_URI}?success=1&login={login}"
    return RedirectResponse(url=redirect_url)

@router.post("/x/disconnect")
async def disconnect_x_account(user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    """
    Unlink X (Twitter) account from user
    """
    user.twitter_login = None
    db.commit()
    return {"message": "X account disconnected"}


class Multiplier(BaseModel):
    value: float
    multiplier: float
    
class MindshareTable(BaseModel):
    current: float
    today: float
    yesterday: float

class PersonalResultsResponse(BaseModel):
    total_score: float
    mindshare: MindshareTable
    loyalty_bonus: Multiplier # todo: разобраться с этим параметром
    streak_bonus: Multiplier
    new_author_bonus: Multiplier
    total_aura_score: float = 0.0  # Общее количество aura очков

def _calc_mindshare(payouts, start_ts, end_ts) -> float:
    """
    Для заданного периода считает средневзвешенный mindshare пользователя.
    Аналогично build_leaderboard_users, но только для одного social_account.
    """
    if not payouts:
        return 0.0
    
    all_period_weight = end_ts - start_ts
    user_mindshare = 0
    logger.info(f"all_period_weight: {all_period_weight}")
    for payout in payouts:
        h = payout.project_leaderboard_history
        # Период history
        if h.start_ts >= end_ts or h.end_ts <= start_ts:
            # пропускаем, если период не пересекается с заданным
            logger.info(f"skipping payout: {payout}")
            continue
        period_weight = min(h.end_ts, end_ts) - max(h.start_ts, start_ts)
        logger.info(f"payperiod_weightout: {period_weight} end_ts: {end_ts} start_ts: {start_ts}")
        user_mindshare += (payout.mindshare * period_weight) / all_period_weight
        logger.info(f"user_mindshare: {user_mindshare}")
        
    return user_mindshare
    
@router.get("/leaderboard/personal", response_model=PersonalResultsResponse)
async def get_personal_results(
    project_id: int = Query(..., description="ID проекта для персональных результатов (обязательный параметр)"),
    user: models.User = Depends(get_current_user), 
    db: Session = Depends(get_session)
):
    """
    Получение персональных результатов пользователя для указанного проекта лидерборда
    """
    # 1. Проверяем наличие twitter_login
    if not user.twitter_login:
        raise exceptions.XAccountNotLinkedError()

    # 2. Получаем проект для лидерборда
    project = db.query(models.Project).filter(
        models.Project.id == project_id,
        models.Project.is_leaderboard_project == True
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"Leaderboard project with id '{project_id}' not found")

    # 3. Получаем social_account по twitter_login
    social_account = db.query(models.SocialAccount).filter(models.SocialAccount.social_login == user.twitter_login).first()
    #social_account = db.query(models.SocialAccount).filter(models.SocialAccount.social_login == "Calderaxyz").first()
    if not social_account:
        # такого быть не должно, но если так, то возвращаем 0
        logger.error(f"User {user.id} has no social_account")
        return PersonalResultsResponse(
            total_score=0,
            mindshare=MindshareTable(current=0, today=0, yesterday=0),
            loyalty_bonus=Multiplier(value=0, multiplier=1.0),
            streak_bonus=Multiplier(value=0, multiplier=1.0),
            new_author_bonus=Multiplier(value=0, multiplier=1.0),
            total_aura_score=0.0
        )

    # 4. Получаем все payouts по этому проекту и social_account
    payouts = await crud.get_account_payouts(db, project.id, social_account.id)
    payouts.sort(key=lambda p: p.project_leaderboard_history.created_at)
    if not payouts:
        logger.error(f"User {user.id}(social_account_id={social_account.id}) has no payouts for project {project.id}")
        return PersonalResultsResponse(
            total_score=0,
            mindshare=MindshareTable(current=0, today=0, yesterday=0),
            loyalty_bonus=Multiplier(value=0, multiplier=1.0),
            streak_bonus=Multiplier(value=0, multiplier=1.0),
            new_author_bonus=Multiplier(value=0, multiplier=1.0),
            total_aura_score=0.0
        )

    # Считаем не ровно день назад, а за чуть меньший промежуток времени ( не учитываем время с последнего обновления)
    end_ts = await crud.get_project_leaderboard_last_ts(db, project.id)
    day_seconds = 86400
    now_ts = utils_base.now_timestamp()
    today_mindshare = _calc_mindshare(payouts, now_ts - day_seconds, end_ts)
    yesterday_mindshare = _calc_mindshare(payouts, now_ts - 2*day_seconds, end_ts - day_seconds)
    
    all_time_mindshare = _calc_mindshare(payouts, 0, end_ts)
    logger.info(f"all_time_mindshare: {all_time_mindshare}")
    
    # отображаем мультипликаторы по последнему payout
    last_payout = payouts[-1]
    # TODO: правильно считать мультипликаторы
    # 1. Loyalty
    user_actual_loyalty = last_payout.loyalty_points
    user_last_plan_check = crud.get_user_last_plan_check(db, user.id)
    if user_last_plan_check:
        if user_last_plan_check.user_plan == enums.SubscriptionPlanType.ULTRA:
            user_actual_loyalty += 30
    user_actual_loyalty = min(user_actual_loyalty, 100)
    logger.info(f"user_actual_loyalty: {user_actual_loyalty}")
    loyalty_bonus = Multiplier(value=user_actual_loyalty, multiplier=utils_base.loyalty_to_multiplier(user_actual_loyalty))
    # 2. Streak
    streak_is_active = now_ts - last_payout.last_post_at > day_seconds*7
    if streak_is_active:
        # вычисляем текущую серию
        streak = (now_ts - last_payout.weekly_streak_start_at)//(day_seconds*7)
    else:
        streak = 0
    streak_bonus = Multiplier(value=streak, multiplier=utils_base.streak_to_multiplier(streak))
    # 3. New author bonus
    new_author_bonus = now_ts - last_payout.first_post_at < day_seconds*7
    new_author_bonus = Multiplier(value=int(new_author_bonus), multiplier=(10 if new_author_bonus else 1))
    # TODO: проблема, что пок какой-то причине можем потерять стрики человека.
    streak_bonus = Multiplier(value=(now_ts - last_payout.weekly_streak_start_at)//(day_seconds*7), multiplier=1.0)
    new_author_bonus = now_ts - last_payout.first_post_at < day_seconds*7
    new_author_bonus = Multiplier(value=int(new_author_bonus), multiplier=(10 if new_author_bonus else 1))
    
    # Получаем общее количество aura очков для этого социального аккаунта
    total_aura = db.query(models.PostAuraScore).filter(
        models.PostAuraScore.social_account_id == social_account.id
    ).all()
    total_aura_score = sum(score.aura_score for score in total_aura)

    return PersonalResultsResponse(
        total_score=sum(p.score for p in payouts),
        mindshare=MindshareTable(
            current=payouts[-1].mindshare, 
            today=today_mindshare, 
            yesterday=yesterday_mindshare
        ),
        loyalty_bonus=loyalty_bonus,
        streak_bonus=streak_bonus,
        new_author_bonus=new_author_bonus,
        total_aura_score=total_aura_score
    )
    
@router.post("/og-bonus")
async def activate_og_bonus(user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    """
    Активация бонуса за ОГ
    """
    if user.og_bonus_activated:
        return {"success": True, "message": "already_activated"}
    
    user.og_bonus_activated = True
    db.commit()
    return {"success": True, "message": "activated"}


@router.get("/aura/history", response_model=schemas.GetAuraScoresResponse)
async def get_user_aura_history(
    project_id: int = Query(..., description="ID проекта для фильтрации истории Aura (обязательный параметр)"),
    user: models.User = Depends(get_current_user), 
    db: Session = Depends(get_session)
):
    """
    Получение истории aura scores пользователя для указанного проекта
    """
    # 1. Проверяем наличие project_id
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id is required")
    
    # 2. Проверяем существование проекта
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"Project with id '{project_id}' not found")

    # 3. Проверяем наличие twitter_login
    if not user.twitter_login:
        raise exceptions.XAccountNotLinkedError()

    # 4. Получаем social_account по twitter_login
    social_account = db.query(models.SocialAccount).filter(
        models.SocialAccount.social_login == user.twitter_login
    ).first()
    if not social_account:
        logger.error(f"User {user.id} has no social_account")
        return schemas.GetAuraScoresResponse(aura_scores=[], total_aura_score=0.0)

    # 5. Получаем все aura scores для этого социального аккаунта для указанного проекта
    query = db.query(models.PostAuraScore).filter(
        models.PostAuraScore.social_account_id == social_account.id,
        models.PostAuraScore.project_id == project_id
    )
    
    aura_scores = query.order_by(models.PostAuraScore.created_at.desc()).all()

    # 6. Преобразуем в схему
    aura_records = []
    total_score = 0.0
    
    for score in aura_scores:
        aura_records.append(schemas.AuraScoreRecord(
            post_full_text=score.post_full_text,
            created_at=score.created_at,
            aura_score=score.aura_score,
            project_id=score.project_id,
            project_name=project.name
        ))
        total_score += score.aura_score

    return schemas.GetAuraScoresResponse(
        aura_scores=aura_records,
        total_aura_score=total_score
    )

