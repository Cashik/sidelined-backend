from pydantic_settings import BaseSettings
from functools import lru_cache
import os
from pydantic import ConfigDict, validator
from src import enums, schemas


project_tokens = {
    "RIZ": schemas.Token(
        chain_id=enums.ChainID.BASE,
        address="0x67543CF0304C19CA62AC95ba82FD4F4B40788dc1",
        interface=enums.TokenInterface.ERC20,
        decimals=8,
        symbol="RIZ",
        name="Rivals Network"
    ),
    "wRIZ": schemas.Token(
        chain_id=enums.ChainID.BASE,
        address="0xA70acF9Cbb8CA5F6c2A9273283fb17C195ab7a43",
        interface=enums.TokenInterface.ERC20,
        decimals=8,
        symbol="wRIZ",
        name="Staked RIZ"
    ),
    "ZNL": schemas.Token(
        chain_id=enums.ChainID.ARBITRUM,
        address="0x78bDE7b6C7eB8f5F1641658c698fD3BC49738367",
        interface=enums.TokenInterface.ERC721,
        decimals=0,
        symbol="ZNL",
        name="ZNodeLicense"
    )
}


pro_plan_requirements = [
    schemas.TokenRequirement(
        token=project_tokens["RIZ"],
        balance=1000
    ),
    schemas.TokenRequirement(
        token=project_tokens["wRIZ"],
        balance=1000
    ),

]

ultra_plan_requirements = [
    schemas.TokenRequirement(
        token=project_tokens["RIZ"],
        balance=8000
    ),
    schemas.TokenRequirement(
        token=project_tokens["wRIZ"],
        balance=8000
    ),
    schemas.TokenRequirement(
        token=project_tokens["ZNL"],
        balance=1
    )
]

available_models = [
    schemas.AIModel(
        id=enums.Model.GPT_4O,
        provider=enums.Service.OPENAI,
        name="GPT-4O",
        description="Fast, intelligent, flexible GPT model"
    ),
    schemas.AIModel(
        id=enums.Model.GPT_4_1,
        provider=enums.Service.OPENAI,
        name="GPT-4.1",
        description="Flagship GPT model for complex tasks"
    ),
    schemas.AIModel(
        id=enums.Model.GPT_O4_MINI,
        provider=enums.Service.OPENAI,
        name="o4-mini",
        description="Fast reasoning model"
    ),
    schemas.AIModel(
        id=enums.Model.GEMINI_2_5_PRO,
        provider=enums.Service.GEMINI,
        name="gemini-2.5-pro-preview-03-25",
        description="Intelligent and flexible Gemini model"
    ),
    schemas.AIModel(
        id=enums.Model.GEMINI_2_5_FLASH,
        provider=enums.Service.GEMINI,
        name="gemini-2.5-flash-preview-04-17",
        description="Fast and flexible Gemini model"
    )
]


all_ai_models_ids = list(enums.Model)
basic_ai_models_ids = [enums.Model.GPT_4O, enums.Model.GPT_O4_MINI, enums.Model.GEMINI_2_5_FLASH]
pro_ai_models_ids = [enums.Model.GPT_4_1, enums.Model.GEMINI_2_5_PRO]

subscription_plans = [
    schemas.SubscriptionPlanExtended(
        id=enums.SubscriptionPlanType.BASIC,
        name="Basic",
        requirements=[],
        max_credits=30,
        available_models_ids=basic_ai_models_ids,
        available_tools=[]
    ),
    schemas.SubscriptionPlanExtended(
        id=enums.SubscriptionPlanType.PRO,
        name="Pro",
        requirements=pro_plan_requirements,
        max_credits=100,
        available_models_ids=pro_ai_models_ids+basic_ai_models_ids,
        available_tools=[]
    ),
    schemas.SubscriptionPlanExtended(
        id=enums.SubscriptionPlanType.ULTRA,
        name="Ultra",
        requirements=ultra_plan_requirements,
        max_credits=10000,
        available_models_ids=all_ai_models_ids,
        available_tools=[]
    )
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
    TOKEN_REQUIREMENTS: list[schemas.TokenRequirement] = pro_plan_requirements
    SUBSCRIPTION_PLANS: list[schemas.SubscriptionPlan] = subscription_plans
    BALANCE_CHECK_LIFETIME_SECONDS: int = 60*60*4 # default 4 hours
    
    # MCP Servers and tools API keys
    EVM_AGENT_KIT_SSE_URL: str | None = None
    SMITHERY_API_KEY: str | None = None
    EXA_SEARCH_API_KEY: str | None = None
    
    ANKR_API_KEY: str
    
    # todo: валидировать планы подписки
    @validator('TOKEN_REQUIREMENTS')
    def validate_token_requirements(cls, v):
        for req in v:
            if req.token.interface not in [enums.TokenInterface.ERC20, enums.TokenInterface.ERC721]:
                raise NotImplementedError(
                    f"Currently only ERC20 and ERC721 tokens are supported. "
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


