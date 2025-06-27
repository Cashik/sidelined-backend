from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Body
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from src import schemas, enums, models, crud, utils, exceptions
from src.core.crypto import verify_signature, validate_payload, SOLANA_NETWORKS_IDS
from src.core.middleware import get_current_user
from src.database import get_session
from src.config.settings import settings
from src.exceptions import (
    AddressAlreadyExistsError, AddressNotFoundError, LastAddressError, FactNotFoundError, APIError
)

router = APIRouter(prefix="/user", tags=["User"])

class UserProfileUpdateRequest(BaseModel):
    preferred_name: Optional[str] = None
    user_context: Optional[str] = None

class ChatSettingsUpdateRequest(schemas.UserChatSettings):
    pass


class UserFactDeleteRequest(BaseModel):
    id: int

class UserFactAddRequest(BaseModel):
    description: str

class WalletAddressAddRequest(BaseModel):
    payload: schemas.LoginPayload
    signature: str

class WalletAddressDeleteRequest(BaseModel):
    address: str


def db_user_to_schema_user(user: models.User) -> schemas.User:
    facts = [schemas.UserFact(id=fact.id, description=fact.description, created_at=fact.created_at) for fact in user.facts]
    wallet_addresses = [schemas.WalletAddress(address=wallet.address, created_at=wallet.created_at) for wallet in user.wallet_addresses]
    if user.chat_settings:
        chat_settings = schemas.MessageGenerationSettings.model_validate(user.chat_settings)
    else:
        chat_settings = schemas.MessageGenerationSettings()
    
    return schemas.User(
        profile=schemas.UserProfile(
            preferred_name=user.preferred_name, 
            user_context=user.user_context,
            facts=facts
        ),
        chat_settings=schemas.UserChatSettings(
            preferred_chat_model=chat_settings.model,
            preferred_chat_style=chat_settings.chat_style, 
            preferred_chat_details_level=chat_settings.chat_details_level
        ),
        connected_wallets=wallet_addresses,
        x_login=user.twitter_login,
        og_bonus_activated=user.og_bonus_activated
    )


@router.get("/me", response_model=schemas.User)
async def get_user(user: models.User = Depends(get_current_user)):
    return db_user_to_schema_user(user)


@router.post("/settings/chat", response_model=schemas.UserChatSettings)
async def update_chat_settings(request: ChatSettingsUpdateRequest, user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    new_chat_settings = schemas.MessageGenerationSettings(
        model=request.preferred_chat_model,
        chat_style=request.preferred_chat_style,
        chat_details_level=request.preferred_chat_details_level
    )
    user_data: models.User = await crud.update_user_chat_settings(user.id, new_chat_settings, db)
    user_chat_settings_response = schemas.UserChatSettings()
    if user_data.chat_settings:
        user_chat_settings = schemas.MessageGenerationSettings.model_validate(user_data.chat_settings)
        user_chat_settings_response.preferred_chat_model = user_chat_settings.model
        user_chat_settings_response.preferred_chat_style = user_chat_settings.chat_style
        user_chat_settings_response.preferred_chat_details_level = user_chat_settings.chat_details_level
    return user_chat_settings_response


@router.post("/settings/profile", response_model=schemas.User)
async def update_user_settings(request: UserProfileUpdateRequest, user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    user_data: models.User = await crud.update_user_profile(user, request, db)
    return db_user_to_schema_user(user_data)


@router.post("/facts/add", response_model=schemas.User)
async def add_user_fact(request: UserFactAddRequest, user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    user_data: models.User = await crud.add_user_facts(user, [request.description], db)
    return db_user_to_schema_user(user_data)


@router.post("/facts/delete", response_model=schemas.User)
async def delete_user_fact(request: UserFactDeleteRequest, user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    try:
        user_data: models.User = await crud.delete_user_facts(user, [request.id], db)
        return db_user_to_schema_user(user_data)
    except FactNotFoundError as e:
        raise APIError(code=e.code, message=e.message, status_code=400)
    except Exception as e:
        raise


@router.post("/wallet/add", response_model=schemas.WalletAddress)
async def add_wallet_address(
    request: WalletAddressAddRequest,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Добавление нового адреса кошелька пользователю
    
    Проверяет подпись и добавляет адрес, если она валидна
    """
    if request.payload.chain_id in SOLANA_NETWORKS_IDS:
        chain_family = enums.ChainFamily.SOLANA
    else:
        chain_family = enums.ChainFamily.EVM
    
    # Проверяем валидность payload'а
    if not validate_payload(request.payload):
        raise APIError(code="invalid_payload", message="Invalid payload", status_code=400)
    
    # Проверяем подпись
    if not verify_signature(request.payload, request.signature):
        raise APIError(code="invalid_signature", message="Invalid signature", status_code=401)
    
    try:
        # Добавляем адрес
        wallet = await crud.add_user_address(
            user=user,
            address=request.payload.address,
            chain_family=chain_family,
            session=db
        )
        return schemas.WalletAddress(
            id=wallet.id,
            address=wallet.address,
            created_at=wallet.created_at
        )
    except AddressAlreadyExistsError as e:
        raise APIError(code=e.code, message=e.message, status_code=400)
    except Exception as e:
        raise


@router.post("/wallet/delete")
async def delete_wallet_address(
    request: WalletAddressDeleteRequest,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Удаление адреса кошелька пользователя
    
    Нельзя удалить последний адрес пользователя
    """
    try:
        await crud.delete_user_address(
            user=user,
            address=request.address,
            session=db
        )
        return {"status": "success"}
    except AddressNotFoundError as e:
        raise APIError(code=e.code, message=e.message, status_code=404)
    except LastAddressError as e:
        raise APIError(code=e.code, message=e.message, status_code=400)
    except Exception as e:
        raise






