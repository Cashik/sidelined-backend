import jwt
import time
from typing import Optional, Dict, Any

from src.config import settings
from src.schemas import TokenPayload

def create_token(payload: TokenPayload, expires_delta: Optional[int] = None) -> str:
    """
    Создает JWT токен для пользователя с информацией о проверке баланса
    """
    to_encode = payload.model_dump()
    jwt_settings = settings.get_jwt_settings()
    current_time = int(time.time())
    expiration_time = current_time + (expires_delta or jwt_settings["access_token_expire_minutes"] * 60)
    to_encode.update({
        "exp": expiration_time,
        "iat": current_time
    })
    
    encoded_jwt = jwt.encode(
        to_encode, 
        jwt_settings["secret_key"], 
        algorithm=jwt_settings["algorithm"]
    )
    return encoded_jwt


def decode_token(token: str) -> Dict[str, Any]:
    try:
        jwt_settings = settings.get_jwt_settings()
        payload = jwt.decode(
            token, 
            jwt_settings["secret_key"], 
            algorithms=[jwt_settings["algorithm"]]
        )
        return payload
    except jwt.PyJWTError as e:
        # В реальном приложении здесь можно логировать ошибки
        raise e
