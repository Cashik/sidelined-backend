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
from src.core.middleware import get_current_user, check_balance_and_update_token
from src.database import get_session
from src.services import user_context_service
from src.config.settings import settings
from src.config.mcp_servers import get_toolboxes, mcp_servers
from src.services.prompt_service import PromptService
from src.config import ai_models, subscription_plans
from src.services.x_api_service import XApiService
from src.services.x_oauth_service import XOAuthService

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
    include_project_sources: bool = True,
    sort_type: enums.SortType = enums.SortType.POPULAR,
    db: Session = Depends(get_session)
):
    """
    Получение ленты постов, связанных с проектом (по упоминаниям), с возможностью исключения постов от связанных аккаунтов.
    """
    if not projects_ids:
        projects = await crud.get_projects_all(db)
        projects_ids = [p.id for p in projects]
    posts = await crud.get_project_feed_posts(
        db=db,
        project_ids=projects_ids,
        include_project_sources=include_project_sources,
        sort_type=sort_type,
        limit=100,
    )
    posts_schemas = utils.convert_posts_to_schemas(posts)
    return schemas.GetFeedResponse(posts=posts_schemas)

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
async def personalize(request: YapsPersonalizationRequest, user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    """
    Персонализация текста для авто-постов
    """
    # TODO: снимать кредиты с пользователя
    # Получаем настройки персонализации пользователя
    personalization_settings = await crud.get_brain_settings(user, db)
    
    # Генерируем 3 персонализированных варианта
    variants = await utils.generate_personalized_tweets(
        original_text=request.text,
        personalization_settings=personalization_settings,
        count=3
    )
    return YapsPersonalizationResponse(text=variants[0], variants=variants)

    
class LeaderboardResponse(BaseModel):
    users: List[schemas.LeaderboardUser]
    

@router.get("/leaderboard", response_model=LeaderboardResponse)
async def get_leaderboard(days: int = Query(1, ge=1, le=7, description="Number of days to get leaderboard for"), db: Session = Depends(get_session)):
    """
    Получение лидерборда за определенное количество дней (1, 3 или 7 дней)
    """
    # 1. Получаем первый проект с is_leaderboard_project=True
    project = db.query(models.Project).filter(models.Project.is_leaderboard_project == True).first()
    if not project:
        return LeaderboardResponse(users=[])

    # 2. Получаем истории за период
    now_ts = int(time.time())
    from_ts = now_ts - days * 86400
    histories = (
        db.query(models.ProjectLeaderboardHistory)
        .options(joinedload(models.ProjectLeaderboardHistory.scores).joinedload(models.ScorePayout.social_account))
        .filter(models.ProjectLeaderboardHistory.project_id == project.id)
        .filter(models.ProjectLeaderboardHistory.created_at >= from_ts)
        .order_by(models.ProjectLeaderboardHistory.created_at)
        .all()
    )
    if not histories:
        return LeaderboardResponse(users=[])

    # 3. Собираем leaderboard
    users = utils.build_leaderboard_users(histories, from_ts)
    return LeaderboardResponse(users=users)

# --- X (Twitter) OAuth ---

# Временное хранилище state -> user_id (для продакшена лучше Redis с TTL)
OAUTH_STATE_STORAGE = {}

@router.get("/x/connect-url")
async def get_x_connect_url(user: models.User = Depends(get_current_user)):
    """
    Получить ссылку для подключения X (Twitter) аккаунта через OAuth (с state)
    """
    state = str(uuid4())
    OAUTH_STATE_STORAGE[state] = user.id  # В реале лучше хранить с TTL
    oauth = XOAuthService()
    url = oauth.get_authorize_url(state)
    return {"url": url}

@router.get("/x/callback")
async def x_oauth_callback(code: str, state: str, db: Session = Depends(get_session)):
    """
    Callback for OAuth X (Twitter). Saves the user's X login to the User model by state.
    On success, redirects to frontend URL from settings.TWITTER_SUCCESS_REDIRECT_URI.
    """
    user_id = OAUTH_STATE_STORAGE.pop(state, None)
    if not user_id:
        raise HTTPException(status_code=400, detail="Invalid or expired state")
    user = db.get(models.User, user_id)  # FIX: db.get is sync
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
    value: int
    multiplier: float
    
class MindshareTable(BaseModel):
    current: float
    today: float
    yesterday: float

class PersonalResultsResponse(BaseModel):
    total_score: int
    mindshare: MindshareTable
    loyalty_bonus: Multiplier # todo: разобраться с этим параметром
    streak_bonus: Multiplier
    new_author_bonus: Multiplier

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
    
@router.get("/leaderboard/personal")
async def get_personal_results(user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    """
    Получение персональных результатов пользователя
    """
    # 1. Проверяем наличие twitter_login
    if not user.twitter_login:
        raise exceptions.XAccountNotLinkedError()

    # 2. Получаем первый проект с is_leaderboard_project=True
    project = db.query(models.Project).filter(models.Project.is_leaderboard_project == True).first()
    if not project:
        raise HTTPException(status_code=404, detail="There is no leaderboard project.")

    # 3. Получаем social_account по twitter_login
    social_account = db.query(models.SocialAccount).filter(models.SocialAccount.social_login == user.twitter_login).first()
    #social_account = db.query(models.SocialAccount).filter(models.SocialAccount.social_login == "Calderaxyz").first()
    if not social_account:
        # такого быть не должно, но если так, то возвращаем 0
        return PersonalResultsResponse(
            total_score=0,
            mindshare=MindshareTable(current=0, today=0, yesterday=0),
            loyalty_bonus=Multiplier(value=0, multiplier=1.0),
            streak_bonus=Multiplier(value=0, multiplier=1.0),
            new_author_bonus=Multiplier(value=0, multiplier=1.0)
        )

    # 4. Получаем все payouts по этому проекту и social_account
    payouts = await crud.get_account_payouts(db, project.id, social_account.id)
    payouts.sort(key=lambda p: p.project_leaderboard_history.created_at)
    if not payouts:
        return PersonalResultsResponse(
            total_score=0,
            mindshare=MindshareTable(current=0, today=0, yesterday=0),
            loyalty_bonus=Multiplier(value=0, multiplier=1.0),
            streak_bonus=Multiplier(value=0, multiplier=1.0),
            new_author_bonus=Multiplier(value=0, multiplier=1.0)
        )

    # Считаем не ровно день назад, а за чуть меньший промежуток времени ( не учитываем время с последнего обновления)
    end_ts = await crud.get_project_leaderboard_last_ts(db, project.id)
    day_seconds = 86400
    now_ts = utils_base.now_timestamp()
    today_mindshare = _calc_mindshare(payouts, now_ts - day_seconds, end_ts)
    yesterday_mindshare = _calc_mindshare(payouts, now_ts - 2*day_seconds, end_ts - day_seconds)
    
    all_time_mindshare = _calc_mindshare(payouts, 0, end_ts)
    logger.info(f"all_time_mindshare: {all_time_mindshare}")
    
    return PersonalResultsResponse(
        total_score=int(sum(p.score for p in payouts)),
        mindshare=MindshareTable(
            current=payouts[-1].mindshare, 
            today=today_mindshare, 
            yesterday=yesterday_mindshare
        ),
        loyalty_bonus=Multiplier(value=0, multiplier=1.0),
        streak_bonus=Multiplier(value=0, multiplier=1.0),
        new_author_bonus=Multiplier(value=0, multiplier=1.0)
    )
    
class CoinMarketCapResponse(BaseModel):
    price: float
    market_cap: float
    volume: float
    change_24h: float
    change_7d: float

