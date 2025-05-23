from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, Dict, Any, List, Optional, Tuple, Union
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import select, func, desc
import logging
import json
import jsonschema
from jsonschema import validate, ValidationError
import time

from src import schemas, enums, models, crud, utils, utils_base, exceptions
from src.core.middleware import get_current_user, check_balance_and_update_token
from src.database import get_session
from src.services import user_context_service
from src.config.settings import settings
from src.config.mcp_servers import get_toolboxes, mcp_servers
from src.services.prompt_service import PromptService
from src.config import ai_models, subscription_plans
from src.services.x_api_service import XApiService

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter(prefix="/projects", tags=["Auto-Yaps"])


@router.get("/posts/search/test")
async def search_posts(query: str, db: Session = Depends(get_session)):
    """
    Поиск постов в X (Twitter)
    """
    project = models.Project(
        name="query",
        description="test",
        url="test",
        keywords="test"
    )
    # время за неделю назад
    tm = utils_base.now_timestamp() - 60*60*24*7
    await utils.update_project_data(project, tm, db)
    return "ok"

class Project(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    url: Optional[str] = None

class GetProjectsResponse(BaseModel):
    projects: List[Project]


@router.get("/projects/all", response_model=GetProjectsResponse)
async def get_projects(db: Session = Depends(get_session)):
    """
    Получение списка всех отслеживаемых проектов приложения
    """
    projects_db: List[models.Project] = await crud.get_projects_all(db)
    projects = [Project(id=project.id, name=project.name, description=project.description, url=project.url) for project in projects_db]
    return GetProjectsResponse(projects=projects)


@router.get("/projects/selected", response_model=GetProjectsResponse)
async def get_selected_projects(user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    """
    Получение списка отслеживаемых проектов для конкретного пользователя
    """
    projects_db: List[models.Project] = await crud.get_projects_selected_by_user(user, db)
    projects = [Project(id=project.id, name=project.name, description=project.description, url=project.url) for project in projects_db]
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
    include_other_sources: bool = True,
    sort_type: enums.SortType = enums.SortType.NEW,
    db: Session = Depends(get_session)
):
    """
    Эндпоинт для получения ленты постов по фильтру с сортировкой и (todo:пагинацией).
    
    Фильтры:
    - projects_ids: List[int] - список id проектов, если не указан, то учитываются все проекты
    - include_project_sources: bool - включать ли посты от источников, связанных с проектами
    - include_other_sources: bool - включать ли посты от источников, которые не относятся к проектам

    Сортировка:
    - sort_type: Enums.SortType - тип сортировки (по новизне или по популярности)
    
    TODO:Пагинация:
    - page: int - номер страницы
    - page_size: int - количество постов на странице
    """
    filter = schemas.FeedFilter(
        projects_ids=projects_ids,
        include_project_sources=include_project_sources,
        include_other_sources=include_other_sources
    )
    
    sort = schemas.FeedSort(type=sort_type)
    
    posts: List[models.SocialPost] = await crud.get_posts(filter, db)
    
    logger.info(f"Found {len(posts)} posts")
    # сортируем посты по выбранному типу сортировки
    def score_post(post: models.SocialPost) -> float:
        # (views *1 + likes *0.5 + retweets*2)
        try:
            stats: models.SocialPostStatistic = post.statistic[0]
            return stats.views * 1 + stats.likes * 0.5 + stats.retweets * 2
        except:
            logger.error(f"No statistics for post {post.id}")
            return 0
    
    if sort.type == enums.SortType.POPULAR:
        posts.sort(key=score_post, reverse=True)
    elif sort.type == enums.SortType.NEW:
        posts.sort(key=lambda x: x.posted_at, reverse=True)
    else:
        raise Exception("Invalid sort type")
    
    # Преобразуем модели в схемы
    posts_schemas = []
    for post in posts:
        try:
            # Получаем статистику поста
            stats = post.statistic[0] if post.statistic else None
            
            # Получаем статусы аккаунта в проектах
            account_projects_statuses = []
            for project_status in post.account.projects:
                account_projects_statuses.append(schemas.SourceStatusInProject(
                    project_id=project_status.project_id,
                    source_type=project_status.type
                ))
            
            # Создаем схему поста
            post_schema = schemas.Post(
                text=post.text,
                created_timestamp=post.posted_at,
                full_post_json=post.raw_data,  # В данном случае полный текст совпадает с обычным
                stats=schemas.PostStats(
                    favorite_count=stats.likes if stats else 0,
                    retweet_count=stats.reposts if stats else 0,
                    reply_count=stats.comments if stats else 0,
                    views_count=stats.views if stats else 0
                ),
                account_name=post.account.name,
                account_projects_statuses=account_projects_statuses
            )
            posts_schemas.append(post_schema)
        except Exception as e:
            logger.error(f"Error converting post {post.id} to schema: {str(e)}")
            continue
    
    response = schemas.GetFeedResponse(posts=posts_schemas)
    return response


