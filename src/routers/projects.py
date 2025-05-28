from enum import Enum
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, Dict, Any, List, Optional, Tuple, Union
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
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
        include_other_sources=include_other_sources,
    )

    posts: List[models.SocialPost] = await crud.get_posts(
        filter=filter,
        sort_type=sort_type,
        db=db,
        limit=100,
    )

    logger.info(f"Found {len(posts)} posts")
    
    # Преобразуем модели в схемы
    posts_schemas = utils.convert_posts_to_schemas(posts)
    
    
    response = schemas.GetFeedResponse(posts=posts_schemas)
    return response


@router.get("/brain_settings", response_model=schemas.PersonalizationSettings)
async def get_brain_settings(user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    """
    Получение настроек для нейросети
    """
    return await crud.get_brain_settings(user, db)

@router.post("/brain_settings", response_model=schemas.PersonalizationSettings)
async def set_brain_settings(request: schemas.PersonalizationSettings, user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    """
    Установка настроек для нейросети
    """
    return await crud.set_brain_settings(request, user, db)

class FeedTemplatesResponse(BaseModel):
    templates: List[schemas.PostExample]
    new_templates_available: bool

@router.get("/feed/templates", response_model=FeedTemplatesResponse)
async def get_feed_templates(user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    """
    Получение шаблонов для auto-yaps
    """
    return FeedTemplatesResponse(
        templates=await crud.get_feed_templates(user, db),
        new_templates_available=True
    )

@router.post("/feed/templates", response_model=FeedTemplatesResponse)
async def create_feed_template(user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    """
    Создание шаблонов для auto-yaps
    """
    new_templates = await utils.create_user_autoyaps(user, db)
    return FeedTemplatesResponse(
        templates=new_templates,
        new_templates_available=False
    )


