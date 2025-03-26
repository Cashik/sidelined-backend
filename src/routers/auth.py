from fastapi import APIRouter, Request, HTTPException, Depends
import logging
import uuid
import time
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src import crud, schemas
from src.core.crypto import verify_signature, is_valid_address
from src.core.auth import create_token, decode_token
from src.core.middleware import get_current_user, get_optional_user
from src.database import get_session
from src import models

router = APIRouter(prefix="/auth")

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOGIN_STATEMENT = "Войдите в приложение Sidelined AI используя свой кошелек."
DOMAIN = "localhost:8000"



@router.post("/login", response_model=schemas.LoginResponse)
async def do_login(request: schemas.LoginRequest, db: Session = Depends(get_session)):
    logger.info(f"Login data received: {request}")
    
    if not verify_signature(request.payload, request.signature):
        raise HTTPException(status_code=401, detail="Invalid signature")
    if not is_valid_address(request.payload.address):
        raise HTTPException(status_code=401, detail="Invalid address")
    
    user = await crud.get_or_create_user(request.payload.address, request.payload.chain_id, db)
    token = create_token({"user_id": user.id})
    return schemas.LoginResponse(access_token=token)

@router.post("/logout")
async def do_logout():
    logger.info("Logout request received")
    # Здесь будет логика для выхода пользователя
    return {"message": "Logout successful"}

@router.post("/login-payload", response_model=schemas.LoginPayload)
async def get_login_payload(payload_request: schemas.LoginPayloadRequest):
    logger.info(f"Login payload request received with data: {payload_request}")
    # Генерация уникального nonce
    nonce = str(uuid.uuid4())
    # Текущее время в миллисекундах для срока действия
    expiration_time = int(time.time() * 1000) + 1000 * 60 * 5  # 5 минут срок действия
    # Структура payload в соответствии с требованиями thirdweb
    payload = schemas.LoginPayload(
        domain=DOMAIN,
        address=payload_request.address,
        statement=LOGIN_STATEMENT,
        uri=f"http://{DOMAIN}",  # Добавляем URI в формате http://domain
        version="1",
        chain_id=payload_request.chainId,
        nonce=nonce,
        issued_at=time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        expiration_time=time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(expiration_time / 1000)),
    )
    
    logger.info(f"Generated login payload: {payload}")
    return payload

@router.get("/is-logged-in", response_model=schemas.IsLoginResponse)
async def is_logged_in(user: models.User = Depends(get_optional_user)):
    """
    Проверяет, авторизован ли пользователь.
    Использует get_optional_user для получения пользователя из токена.
    
    Returns:
        IsLoginResponse: Объект с информацией о статусе авторизации
    """
    logger.info("Check if user is logged in")
    
    if user:
        logger.info(f"User is logged in: {user.id} ({user.address})")
        return schemas.IsLoginResponse(logged_in=True)
    
    logger.info("User is not logged in")
    return schemas.IsLoginResponse(logged_in=False)
