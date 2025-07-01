from fastapi import Depends, HTTPException, status, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
from typing import Optional
from sqlalchemy.orm import Session

from src.config.settings import settings
from src.core.auth import decode_token, create_token
from src.database import get_session
from src import models, crud, utils, schemas, utils_base, enums

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
        try:
            payload: schemas.AppTokenPayload = decode_token(token)
        except Exception as e:
            logger.error(f"Token decode error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Получаем пользователя из базы данных
        user = await crud.get_user_by_id(payload.user_id, db)
        if user is None:
            logger.warning(f"User with id {payload.user_id} not found")
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
        payload: schemas.AppTokenPayload = decode_token(token)
        user = await crud.get_user_by_id(payload.user_id, db)
        return user
    except Exception as e:
        logger.warning(f"Optional authentication failed: {str(e)}")
        return None
    
