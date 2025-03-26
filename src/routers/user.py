from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from src import schemas, enums, models, crud, utils
from src.core.middleware import get_current_user
from src.database import get_session

router = APIRouter(prefix="/user", tags=["User"])

class UserProfileUpdateRequest(schemas.UserProfile):
    pass

class ChatSettingsUpdateRequest(schemas.UserChatSettings):
    pass

class AvailableSettingsResponse(BaseModel):
    chat_models: List[enums.Model]
    chat_styles: List[enums.ChatStyle]
    chat_details_levels: List[enums.ChatDetailsLevel]

@router.get("/me", response_model=schemas.User)
async def get_user(user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    user_data: models.User = await crud.get_user_by_id(user.id, db)
    return schemas.User(
        address=user_data.address,
        chain_id=user_data.chain_id,
        profile=schemas.UserProfile(
            preffered_name=user_data.preffered_name,
            user_context=user_data.user_context
        ),
        chat_settings=schemas.UserChatSettings(
            preffered_chat_model=user_data.preffered_chat_model,
            preffered_chat_style=user_data.preffered_chat_style,
            preffered_chat_details_level=user_data.preffered_chat_details_level
        )
    )

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
        preffered_chat_model=user_data.preffered_chat_model,
        preffered_chat_style=user_data.preffered_chat_style,
        preffered_chat_details_level=user_data.preffered_chat_details_level
    )

@router.post("/settings/profile", response_model=schemas.User)
async def update_user_settings(request: UserProfileUpdateRequest, user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    user_data: models.User = await crud.update_user_profile(user.id, request, db)
    return schemas.User(
        address=user_data.address,
        chain_id=user_data.chain_id,
        profile=schemas.UserProfile(
            preffered_name=user_data.preffered_name,
            user_context=user_data.user_context
        ),
        chat_settings=schemas.UserChatSettings(
            preffered_chat_model=user_data.preffered_chat_model,
            preffered_chat_style=user_data.preffered_chat_style,
            preffered_chat_details_level=user_data.preffered_chat_details_level
        )
    )










