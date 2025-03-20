from pydantic_settings import BaseSettings
from functools import lru_cache
import os
from pydantic import ConfigDict

"""
Help: any settings can be set in environment variables through .env file with same name as in class

Example:
HOST=0.0.0.0
PORT=8000
DEBUG=True
etc...
"""

class Settings(BaseSettings):
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Флаг для тестового окружения
    TESTING: bool = False
    
    # Database settings
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "postgres"
    DB_HOST: str = "2eden_db"
    DB_PORT: str = "5432"
    DB_NAME: str = "2eden"
    
    # App settings
    SECRET_KEY: str = "your-secret-key"
    DEBUG: bool = False
    
    # JWT settings
    JWT_SECRET_KEY: str = "your-secret-key"  # Default use main SECRET_KEY
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30  # Default 30 minutes
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7  # For future implementation refresh tokens
    
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000,http://77.73.132.142,http://77.73.132.142:3000,http://alpha.outerworldagentz.ai,https://alpha.outerworldagentz.ai,ws://alpha.outerworldagentz.ai,wss://alpha.outerworldagentz.ai"
    MAX_DAILY_QUESTIONS: int = 3

    # OpenAI and AI settings
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    MAX_DAILY_MESSAGES: int = 10
    
    @property
    def DATABASE_URL(self) -> str:
        """
        Get database url in SQLAlchemy format
        For testing returns in-memory SQLite
        """
        if self.TESTING:
            return "sqlite://"  # Use synchronous SQLite for testing
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    @property
    def ALLOWED_ORIGINS(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    model_config = ConfigDict(
        extra="allow",  # Allow additional fields
        from_attributes=True  # This replaces orm_mode
    )

    def get_jwt_settings(self) -> dict:
        """
        Get JWT settings in dictionary format
        """
        return {
            "secret_key": self.JWT_SECRET_KEY or self.SECRET_KEY,
            "algorithm": self.JWT_ALGORITHM,
            "access_token_expire_minutes": self.JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
            "refresh_token_expire_days": self.JWT_REFRESH_TOKEN_EXPIRE_DAYS
        }

@lru_cache()
def get_settings():
    return Settings()

settings = Settings()


