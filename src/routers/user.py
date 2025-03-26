from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from src import schemas, enums, models, crud, utils
from src.core.middleware import get_current_user
from src.database import get_session

router = APIRouter(prefix="/user", tags=["User"])

class UserProfile(BaseModel):
    preffered_name: Optional[str] = None
    user_context: Optional[str] = None

class UserChatSettings(BaseModel):
    preffered_chat_model: Optional[enums.Model] = None
    preffered_chat_style: Optional[enums.ChatStyle] = None
    preffered_chat_details_level: Optional[enums.ChatDetailsLevel] = None

class User(BaseModel):
    address: str
    chain_id: int
    profile: UserProfile
    chat_settings: UserChatSettings


class UserProfileUpdateRequest(UserProfile):
    pass

class ChatSettingsUpdateRequest(UserChatSettings):
    pass

class AvailableSettingsResponse(BaseModel):
    chat_models: List[enums.Model]
    chat_styles: List[enums.ChatStyle]
    chat_details_levels: List[enums.ChatDetailsLevel]

@router.get("/me", response_model=User)
async def get_user(user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    user_data: models.User = await crud.get_user_by_id(user.id, db)
    return User(
        address=user_data.address,
        chain_id=user_data.chain_id,
        profile=UserProfile(
            preffered_name=user_data.preffered_name,
            user_context=user_data.user_context
        ),
        chat_settings=UserChatSettings(
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


@router.post("/settings/chat", response_model=UserChatSettings)
async def update_chat_settings(request: ChatSettingsUpdateRequest, user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    user_data: models.User = await crud.update_user_chat_settings(user.id, request, db)
    return UserChatSettings(
        preffered_chat_model=user_data.preffered_chat_model,
        preffered_chat_style=user_data.preffered_chat_style,
        preffered_chat_details_level=user_data.preffered_chat_details_level
    )

@router.post("/settings/profile", response_model=User)
async def update_user_settings(request: UserProfileUpdateRequest, user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    user_data: models.User = await crud.update_user_profile(user.id, request, db)
    return User(
        address=user_data.address,
        chain_id=user_data.chain_id,
        preffered_name=user_data.preffered_name,
        user_context=user_data.user_context,
        preffered_chat_style=user_data.preffered_chat_style,
        preffered_chat_details_level=user_data.preffered_chat_details_level
    )










