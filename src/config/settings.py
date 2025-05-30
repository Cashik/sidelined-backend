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
    DATABASE_URL: str
    
    # Redis settings
    REDIS_URL: str
    # Redis SSL settings (для managed Redis на облачных провайдерах)
    REDIS_SSL_CERT_REQS: str = "CERT_NONE"  # CERT_NONE, CERT_OPTIONAL, CERT_REQUIRED
    REDIS_SSL_CHECK_HOSTNAME: bool = False
    
    # App settings
    SECRET_KEY: str = "your-secret-key"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000,http://77.73.132.142,http://77.73.132.142:3000,http://localhost:5173,http://127.0.0.1:5173"
    
    # JWT settings
    JWT_SECRET_KEY: str = "your-secret-key"  # Default use main SECRET_KEY
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60*24  # Default 24 hours
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7  # For future implementation refresh tokens
    
    # AI providers keys and settings
    OPENAI_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    
    # Chat settings
    DEFAULT_AI_MODEL_ID: enums.Model = enums.Model.GPT_4O
    
    
    FACTS_FUNCTIONALITY_ENABLED: bool = True
    
    # Auto-Yap settings
    POST_SYNC_UPDATE_PERIOD_SECONDS: int = 60*10 # период между синхронизациями данных с твиттером
    POST_SYNC_PERIOD_SECONDS: int = 60*60*24 # период за который будут синхронизироваться посты (т.е. не старше 24 часов)
    POST_SYNC_LIKES_COUNT_MINIMAL: int = 50 # минимальное количество лайков при запросе к твиттеру
    POST_SYNC_MINIMAL_ENGAGEMENT_SCORES: int = 50 # минимальный порог вовлеченности для добавления в базу
    
    POST_FEED_MINIMAL_ENGAGEMENT_SCORES: int = 200 # минимальный порог вовлеченности поста для выдачи пользователю
    
    # Период времени, через который посты станут неактуальными и их нужно будет удалить
    POST_INACTIVE_TIME_SECONDS: int = 60*60*24
    # Период времени, через который система будет очищать старые посты
    POST_CLEANUP_TIME_SECONDS: int = 60*60*24
    # Период времени, через который система будет создавать авто-yaps (шаблоны постов)
    AUTOYAPS_SYNC_PERIOD_SECONDS: int = 60*30  # каждые 30 минут
    
    
    # Token requirements
    BALANCE_CHECK_LIFETIME_SECONDS: int = 60*60*4 # default 4 hours
    
    # MCP Servers and tools API keys
    EVM_AGENT_KIT_SSE_URL: str | None = None
    SMITHERY_API_KEY: str | None = None
    
    ANKR_API_KEY: str
    
    X_RAPIDAPI_KEY: str
    X_TWITTER_API_KEY: str
    
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


