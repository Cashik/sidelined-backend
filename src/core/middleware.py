from fastapi import Depends, HTTPException, status, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
from typing import Optional
from sqlalchemy.orm import Session

from src.config import settings
from src.core.auth import decode_token, create_token
from src.database import get_session
from src import models, crud, utils, schemas, utils_base

logger = logging.getLogger(__name__)

# Используем HTTPBearer для автоматической проверки заголовка Authorization
security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_session)
) -> Optional[models.User]:
    """
    Зависимость для получения текущего пользователя из JWT токена.
    
    Автоматически проверяет заголовок Authorization в запросе и 
    извлекает пользователя из базы данных на основе user_id из токена.
    
    Args:
        credentials: Учетные данные из заголовка Authorization (Bearer token)
        db: Сессия базы данных
        
    Returns:
        Объект пользователя, если токен валидный
        
    Raises:
        HTTPException: Если токен невалидный или пользователь не найден
    """
    try:
        # Получаем токен из заголовка
        token = credentials.credentials
        
        # Декодируем JWT токен
        payload = decode_token(token)
        
        # Проверяем наличие user_id в токене
        if "user_id" not in payload:
            logger.warning("Token doesn't contain user_id")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token - missing user_id",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Получаем пользователя из базы данных
        user_id = payload["user_id"]
        user = await crud.get_user_by_id(user_id, db)
        
        if user is None:
            logger.warning(f"User with id {user_id} not found")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        # Получаем первый адрес кошелька для логирования
        address = user.wallet_addresses[0].address if user.wallet_addresses else "no address"
        logger.info(f"Authenticated user: {user.id} ({address})")
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Опциональная версия зависимости, которая не вызывает исключение, если токен отсутствует
async def get_optional_user(
    request: Request,
    db: Session = Depends(get_session)
) -> Optional[models.User]:
    """
    Зависимость для получения текущего пользователя из JWT токена,
    но без выбрасывания исключения если токен отсутствует.
    
    Args:
        request: Запрос FastAPI
        db: Сессия базы данных
        
    Returns:
        Объект пользователя, если токен валидный, иначе None
    """
    auth_header = request.headers.get("Authorization")
    
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
        
    try:
        token = auth_header.split(" ")[1]
        payload = decode_token(token)
        
        if "user_id" not in payload:
            return None
            
        user_id = payload["user_id"]
        user = await crud.get_user_by_id(user_id, db)
        
        return user
    except Exception as e:
        logger.warning(f"Optional authentication failed: {str(e)}")
        return None
    

async def check_balance_and_update_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    response: Response = None,
    db: Session = Depends(get_session)
) -> bool:
    """
    Middleware для проверки баланса и обновления токена.
    Проверяет баланс токенов пользователя и обновляет токен при необходимости.
    """
    exception = HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Balance check failed",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token = credentials.credentials
    payload = decode_token(token)
    
    # проверяем последнюю проверку
    check_success: bool = payload["balance_check_success"]
    check_time: int = payload["balance_check_time"]
    current_time: int = utils_base.now_timestamp()
    if check_success and (current_time - check_time) < settings.BALANCE_CHECK_LIFETIME_SECONDS:
        logger.info(f"Balance check successful {payload}")
        return True
    
    # проверка не прошла, значит нужно сделать новую
    user_id: int = payload["user_id"]
    try:
        # Получаем пользователя из базы данных
        user = await crud.get_user_by_id(user_id, db)
        if not user:
            raise exception
            
        balance_check_success: bool = await utils.check_user_access(user)
    except Exception as e:
        logger.error(f"Balance check failed: {str(e)}")
        if settings.ALLOW_CHAT_WHEN_SERVER_IS_DOWN:
            return True
        else:
            raise exception
            
    logger.info(f"Balance check fail. New balance check result: {balance_check_success} for user {payload}")
    if balance_check_success:
        # создаем новый токен
        new_token_payload = schemas.TokenPayload(
            user_id=user_id,
            balance_check_time=current_time,
            balance_check_success=balance_check_success
        )
        new_token = create_token(new_token_payload)
        response.headers["X-New-Token"] = new_token
        return True
    else:
        raise exception
        
