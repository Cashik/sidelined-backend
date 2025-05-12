from fastapi import Depends, HTTPException, status, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
from typing import Optional
from sqlalchemy.orm import Session

from src.config import settings
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
    

async def check_balance_and_update_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    response: Response = None,
    db: Session = Depends(get_session)
) -> enums.SubscriptionPlanType:
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
    try:
        payload: schemas.AppTokenPayload = decode_token(token)
    except Exception as e:
        logger.error(f"Token decode error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # проверяем последнюю проверку
    current_time: int = utils_base.now_timestamp()
    
    check_time: int | None = payload.balance_check_time
    subscription_plan: enums.SubscriptionPlanType | None = payload.subscription
    check_data_in_payload = check_time is not None and subscription_plan is not None
    if check_data_in_payload and (current_time - check_time) < settings.BALANCE_CHECK_LIFETIME_SECONDS:
        logger.info(f"Balance check successful {payload}")
        try:
            return subscription_plan
        except ValueError:
            logger.error(f"Invalid subscription id: {subscription_plan}")
            pass
    
    logger.info(f"Balance check failed. New balance check")
    # проверка не прошла, значит нужно сделать новую
    # Получаем пользователя из базы данных
    user = await crud.get_user_by_id(payload.user_id, db)
    if not user:
        raise exception
    
    try:
        new_sub_plan: enums.SubscriptionPlanType = await utils.check_user_access(user)
    except Exception as e:
        # если проверить не вышло, просто возвращаем базовый доступ юзера
        logger.error(f"Balance check failed: {str(e)}")
        return user.subscription_plan
              
    logger.info(f"New balance check result: {new_sub_plan} for user {user}")
    
    # создаем новый токен
    new_token_payload = schemas.TokenPayload(
        user_id=payload.user_id,
        balance_check_time=current_time,
        subscription=new_sub_plan
    )
    new_token = create_token(new_token_payload)
    response.headers["X-New-Token"] = new_token
    return new_sub_plan
        
