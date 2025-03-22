from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
from sqlmodel import Session
from typing import Optional

from src.core.auth import decode_token
from src.database import get_session
from src import models, crud

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
            
        logger.info(f"Authenticated user: {user.id} ({user.address})")
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
