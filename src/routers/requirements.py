from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from src import schemas, enums, models, crud, utils
from src.core.middleware import get_current_user, check_balance_and_update_token
from src.database import get_session
from src.config import settings

router = APIRouter(prefix="/requirements", tags=["Requirements"])


class AvailableResponse(BaseModel):
    available: bool

class RequirementsResponse(BaseModel):
    requirements: list[schemas.TokenRequirement] 

@router.get("/list", response_model=RequirementsResponse)
async def get_requirements():
    return RequirementsResponse(requirements=settings.TOKEN_REQUIREMENTS)

@router.post("/check", response_model=AvailableResponse)
async def get_available(available_balance: bool = Depends(check_balance_and_update_token)):
    """
    Отдельный эндпоинт, который можно использовать для перепроверки баланса
    """
    return AvailableResponse(available=available_balance)
