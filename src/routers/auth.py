from fastapi import APIRouter, Request, HTTPException, Depends
import logging
import uuid
import time
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src import crud, schemas, utils, utils_base
from src.core.crypto import verify_signature, validate_payload
from src.core.auth import create_token, decode_token
from src.core.middleware import get_current_user, get_optional_user
from src.database import get_session
from src import models
from src.config import settings

router = APIRouter(prefix="/auth")

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOGIN_STATEMENT = "Sign in to Sidelined AI using your wallet with required tokens on balance."
DOMAIN = settings.DOMAIN

def get_supported_chain_ids() -> set[int]:
    """Получает список поддерживаемых chain_id из настроек токенов"""
    return {req.token.chain_id.value for req in settings.TOKEN_REQUIREMENTS}

@router.post("/login", response_model=schemas.LoginResponse)
async def do_login(request: schemas.LoginRequest, db: Session = Depends(get_session)):
    logger.info(f"Login data received: {request}")
    
    # Проверяем валидность payload'а
    if not validate_payload(request.payload):
        raise HTTPException(status_code=400, detail="Invalid payload")
    
    # Проверяем подпись
    if not verify_signature(request.payload, request.signature):
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    # Получаем или создаем пользователя
    user = await crud.get_or_create_user(
        address=request.payload.address,
        session=db
    )
    
    # Проверяем баланс
    balance_check_success = await utils.check_user_access(user)
    
    # Создаем токен
    token_payload = schemas.TokenPayload(
        user_id=user.id,
        balance_check_time=utils_base.now_timestamp(),
        balance_check_success=balance_check_success
    )
    token = create_token(token_payload)
    
    # Обновляем баланс кредитов
    await crud.refresh_user_credits(db, user)
    
    # Логируем информацию о пользователе
    logger.info(f"User is logged in: {user.id} ({user.wallet_addresses[0].address})")
    return schemas.LoginResponse(
        access_token=token,
        token_type="bearer",
        chat_available=balance_check_success
    )

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
    # TODO: сделать срок действия в настройках
    expiration_time = utils_base.now_timestamp() + 60 * 5  # 5 минут срок действия
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
        expiration_time=time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(expiration_time)),
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
        logger.info(f"User is logged in: {user.id} ({user.wallet_addresses})")
        return schemas.IsLoginResponse(logged_in=True)
    
    logger.info("User is not logged in")
    return schemas.IsLoginResponse(logged_in=False)
