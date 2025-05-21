from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
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


@router.get("/posts/search")
async def search_posts(query: str, db: Session = Depends(get_session)):
    """
    Поиск постов в X (Twitter)
    """
    project = models.Project(
        name="test",
        description="test",
        url="test",
        keywords="test"
    )
    # время за час назад
    tm = utils_base.now_timestamp() - 60*60
    await utils.update_project_data(project, tm, db)
    return "ok"


@router.get("/projects/all")
async def get_projects(db: Session = Depends(get_session)):
    """
    Получение списка всех отслеживаемых проектов
    """
    raise NotImplementedError

@router.get("/projects/{id}")
async def get_project(id: int, db: Session = Depends(get_session)):
    """"
    Получение детальной информации по конкретному проекту
    """
    raise NotImplementedError


class GetAccountsRequest(BaseModel):
    project_ids: List[int] | None = None
    account_ids: List[int] | None = None
    from_date: str | None = None
    to_date: str | None = None

@router.post("/accounts")
async def get_accounts(request: GetAccountsRequest, db: Session = Depends(get_session)):
    """
    Получение списка инфлюенсеров по фильтру
    """
    raise NotImplementedError


class GetPostsRequest(BaseModel):
    project_ids: List[int]
    account_ids: List[int]
    from_date: str
    to_date: str

@router.post("/posts")
async def get_posts(request: GetPostsRequest, db: Session = Depends(get_session)):
    """
    Получение списка постов по фильтру
    """
    raise NotImplementedError

