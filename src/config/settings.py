from pydantic_settings import BaseSettings
from functools import lru_cache
import os
from pydantic import ConfigDict, validator
from src import enums, schemas


class Settings(BaseSettings):
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DOMAIN: str = "localhost:8000"  # Домен для аутентификации
    
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
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # JWT settings
    JWT_SECRET_KEY: str = "your-secret-key"  # Default use main SECRET_KEY
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60*24  # Default 24 hours
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7  # For future implementation refresh tokens
    
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000,http://77.73.132.142,http://77.73.132.142:3000,http://localhost:5173,http://127.0.0.1:5173"

    # AI providers keys and settings
    OPENAI_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    
    # Chat settings
    DEFAULT_CREDITS: int = 100  # или любое другое дефолтное значение
    DEFAULT_AI_MODEL: enums.Model = enums.Model.GPT_4O
    
    ALLOW_CHAT_WHEN_SERVER_IS_DOWN: bool = False
    
    FACTS_FUNCTIONALITY_ENABLED: bool = True
    
    # Token requirements
    BALANCE_CHECK_LIFETIME_SECONDS: int = 60*60*4 # default 4 hours
    
    # MCP Servers and tools API keys
    EVM_AGENT_KIT_SSE_URL: str | None = None
    SMITHERY_API_KEY: str | None = None
    EXA_SEARCH_API_KEY: str | None = None
    
    ANKR_API_KEY: str
    
    X_RAPIDAPI_KEY: str = "abfd9abbd9msh74ba5c6e1e2cc26p13e011jsnc2bfcb7f60b1"
    X_TWITTER_API_KEY: str = "2WLXZk9kacMXhHxj2cRg19bsuJ98XiQL0ndQ94kMdciuz%7C1574242047661207552-3jI04wPRb0tVkUfFjR4VzJW19ZnQz3"
    
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

    @property
    def FUNCTIONALITY_ENABLED(self) -> bool:
        return self.FACTS_FUNCTIONALITY_ENABLED

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


