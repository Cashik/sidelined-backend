from pydantic_settings import BaseSettings
from functools import lru_cache
import os
from pydantic import ConfigDict, validator
from src import enums, schemas

requirements = [
    schemas.TokenRequirement(
        token=schemas.Token(
            chain_id=enums.ChainID.BASE,
            address="0x67543CF0304C19CA62AC95ba82FD4F4B40788dc1",
            interface=enums.TokenInterface.ERC20,
            decimals=8,
            symbol="RIZ",
            name="Rivals Network"
        ),
        balance=1000
    ),
    schemas.TokenRequirement(
        token=schemas.Token(
            chain_id=enums.ChainID.BASE,
            address="0xA70acF9Cbb8CA5F6c2A9273283fb17C195ab7a43",
            interface=enums.TokenInterface.ERC20,
            decimals=8,
            symbol="wRIZ",
            name="Staked RIZ"
        ),
        balance=1000
    ),
]




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
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # JWT settings
    JWT_SECRET_KEY: str = "your-secret-key"  # Default use main SECRET_KEY
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60*24  # Default 24 hours
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7  # For future implementation refresh tokens
    
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000,http://77.73.132.142,http://77.73.132.142:3000,http://localhost:5173,http://127.0.0.1:5173"
    MAX_DAILY_QUESTIONS: int = 3

    # AI providers keys and settings
    OPENAI_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    
    # Chat settings
    MAX_DAILY_MESSAGES: int = 10
    DEFAULT_CHAT_CONTEXT_LIMIT_NONCE: int = 15 # не в сообщениях, а в nonce
    DEFAULT_AI_MODEL: enums.Model = enums.Model.GPT_4O
    
    # Thirdweb settings
    THIRDWEB_APP_ID: str
    THIRDWEB_PRIVATE_KEY: str
    ALLOW_CHAT_WHEN_SERVER_IS_DOWN: bool = False
    
    FACTS_FUNCTIONALITY_ENABLED: bool = True
    
    # Token requirements
    TOKEN_REQUIREMENTS: list[schemas.TokenRequirement] = requirements
    BALANCE_CHECK_LIFETIME_SECONDS: int = 60 # default 4 hours
    
    # MCP Servers and tools API keys
    SMITHERY_API_KEY: str | None = None
    EXA_SEARCH_API_KEY: str | None = None
    
    @validator('TOKEN_REQUIREMENTS')
    def validate_token_requirements(cls, v):
        for req in v:
            if req.token.interface != enums.TokenInterface.ERC20:
                raise NotImplementedError(
                    f"Currently only ERC20 tokens are supported. "
                    f"Found {req.token.interface.value} interface in configuration."
                )
        return v
    
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


