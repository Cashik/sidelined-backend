import jwt
from typing import Optional, Dict, Any

from src.config.settings import settings
from src import utils_base, schemas, enums

def create_token(payload: schemas.TokenPayload, expires_delta: Optional[int] = None) -> str:
    """
    Создает JWT токен для пользователя с информацией о проверке баланса
    """
    jwt_settings = settings.get_jwt_settings()
    current_time = utils_base.now_timestamp()
    expiration_time = current_time + (expires_delta or jwt_settings["access_token_expire_minutes"] * 60)
    app_payload = schemas.AppTokenPayload(
        **payload.model_dump(),
        exp=expiration_time,
        iat=current_time
    )
    to_encode = app_payload.model_dump(mode="json")
    encoded_jwt = jwt.encode(
        to_encode, 
        jwt_settings["secret_key"], 
        algorithm=jwt_settings["algorithm"]
    )
    return encoded_jwt


def decode_token(token: str) -> schemas.AppTokenPayload:
    try:
        jwt_settings = settings.get_jwt_settings()
        payload_dict = jwt.decode(
            token, 
            jwt_settings["secret_key"], 
            algorithms=[jwt_settings["algorithm"]]
        )
        return schemas.TokenPayload(**payload_dict)
    except jwt.PyJWTError as e:
        raise e
