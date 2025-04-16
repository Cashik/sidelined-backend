from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from src import schemas, enums, models, crud, utils, exceptions
from src.core.crypto import verify_signature, validate_payload
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

class WalletAddressAddRequest(BaseModel):
    payload: schemas.LoginPayload
    signature: str

class WalletAddressDeleteRequest(BaseModel):
    address: str

def db_user_to_schema_user(user: models.User) -> schemas.User:
    facts = [schemas.UserFact(id=fact.id, description=fact.description, created_at=fact.created_at) for fact in user.facts]
    wallet_addresses = [schemas.WalletAddress(address=wallet.address, created_at=wallet.created_at) for wallet in user.wallet_addresses]
    print(f"user data: {user.chain_id}")
    return schemas.User(
        address=wallet_addresses[0].address,
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
        ),
        wallet_addresses=wallet_addresses
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
    # Проверяем валидность payload'а
    if not validate_payload(request.payload):
        raise HTTPException(status_code=400, detail="Invalid payload")
    
    # Проверяем подпись
    if not verify_signature(request.payload, request.signature):
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    try:
        # Добавляем адрес
        wallet = await crud.add_user_address(
            user_id=user.id,
            address=request.payload.address,
            session=db
        )
        return schemas.WalletAddress(
            id=wallet.id,
            address=wallet.address,
            created_at=wallet.created_at
        )
    except exceptions.AddressAlreadyExistsException:
        raise HTTPException(
            status_code=400,
            detail="Address already exists"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e) if settings.DEBUG else "Internal server error"
        )

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
            user_id=user.id,
            address=request.address,
            session=db
        )
        return {"status": "success"}
    except exceptions.AddressNotFoundException:
        raise HTTPException(
            status_code=404,
            detail="Address not found"
        )
    except exceptions.LastAddressException:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete the last address"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e) if settings.DEBUG else "Internal server error"
        )






