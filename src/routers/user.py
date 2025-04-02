from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from src import schemas, enums, models, crud, utils, exceptions
from src.core.middleware import get_current_user
from src.database import get_session
from src.config import settings
router = APIRouter(prefix="/user", tags=["User"])

class UserProfileUpdateRequest(BaseModel):
    preferred_name: Optional[str] = None
    user_context: Optional[str] = None

class ChatSettingsUpdateRequest(schemas.UserChatSettings):
    pass

class AvailableSettingsResponse(BaseModel):
    chat_models: List[enums.Model]
    chat_styles: List[enums.ChatStyle]
    chat_details_levels: List[enums.ChatDetailsLevel]

class UserFactDeleteRequest(BaseModel):
    id: int

class UserFactAddRequest(BaseModel):
    description: str

def db_user_to_schema_user(user: models.User) -> schemas.User:
    facts = [schemas.UserFact(id=fact.id, description=fact.description, created_at=fact.created_at) for fact in user.facts]
    return schemas.User(
        address=user.address,
        chain_id=user.chain_id,
        profile=schemas.UserProfile(
            preferred_name=user.preferred_name, 
            user_context=user.user_context,
            facts=facts
        ),
        chat_settings=schemas.UserChatSettings(
            preferred_chat_model=user.preferred_chat_model,
            preferred_chat_style=user.preferred_chat_style, 
            preferred_chat_details_level=user.preferred_chat_details_level
        )
    )


@router.get("/me", response_model=schemas.User)
async def get_user(user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    user_data: models.User = await crud.get_user_by_id(user.id, db)
    return db_user_to_schema_user(user_data)

@router.get("/settings/chat", response_model=AvailableSettingsResponse)
async def get_available_settings():
    return AvailableSettingsResponse(
        chat_models=list(enums.Model),
        chat_styles=list(enums.ChatStyle),
        chat_details_levels=list(enums.ChatDetailsLevel)
    )

@router.post("/settings/chat", response_model=schemas.UserChatSettings)
async def update_chat_settings(request: ChatSettingsUpdateRequest, user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    user_data: models.User = await crud.update_user_chat_settings(user.id, request, db)
    return schemas.UserChatSettings(
        preferred_chat_model=user_data.preferred_chat_model,
        preferred_chat_style=user_data.preferred_chat_style,
        preferred_chat_details_level=user_data.preferred_chat_details_level
    )

@router.post("/settings/profile", response_model=schemas.User)
async def update_user_settings(request: UserProfileUpdateRequest, user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    user_data: models.User = await crud.update_user_profile(user.id, request, db)
    return db_user_to_schema_user(user_data)


@router.post("/facts/add", response_model=schemas.User)
async def add_user_fact(request: UserFactAddRequest, user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    user_data: models.User = await crud.add_user_facts(user.id, [request.description], db)
    return db_user_to_schema_user(user_data)


@router.post("/facts/delete", response_model=schemas.User)
async def delete_user_fact(request: UserFactDeleteRequest, user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    try:
        user_data: models.User = await crud.delete_user_facts(user.id, [request.id], db)
        return db_user_to_schema_user(user_data)
    except exceptions.FactNotFoundException:
        raise HTTPException(status_code=400, detail="Fact not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e) if settings.DEBUG else "Internal server error")






