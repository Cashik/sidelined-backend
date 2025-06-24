from typing import Annotated, Any, Callable, Dict, List, Literal, Optional, Union
from decimal import Decimal
from pydantic import BaseModel, Field, ConfigDict, TypeAdapter, field_validator
from datetime import datetime
import time

from src import enums, utils_base
import logging

logger = logging.getLogger(__name__)


# апи-схемы роутера auth

class LoginPayloadRequest(BaseModel):
    address: str
    chainId: int


class LoginPayload(BaseModel):
    domain: str
    address: str
    statement: str
    uri: str
    version: str
    chain_id: int
    nonce: str
    issued_at: str
    expiration_time: str


class LoginRequest(BaseModel):
    payload: LoginPayload
    signature: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    subscription_id: enums.SubscriptionPlanType
    # TODO: убрать chat_available - устарело
    chat_available: bool = True
    


class IsLoginResponse(BaseModel):
    logged_in: bool


# апи-схемы роутера user

class UserFact(BaseModel):
    id: int
    description: str
    created_at: int

class WalletAddress(BaseModel):
    address: str
    created_at: int

class UserProfile(BaseModel):
    preferred_name: Optional[str] = None
    user_context: Optional[str] = None
    facts: List[UserFact]

class UserChatSettings(BaseModel):
    preferred_chat_model: Optional[enums.Model] = None
    preferred_chat_style: Optional[enums.ChatStyle] = None
    preferred_chat_details_level: Optional[enums.ChatDetailsLevel] = None

class User(BaseModel):
    profile: UserProfile
    chat_settings: UserChatSettings
    connected_wallets: List[WalletAddress]
    x_login: Optional[str] = None
    og_bonus_activated: bool = False

# бизнес-схемы токенов

class Token(BaseModel):
    chain_id: enums.ChainID
    address: str
    interface: enums.TokenInterface
    decimals: int
    symbol: str
    name: str

class TokenRequirement(BaseModel):
    token: Token
    # ! количество токенов не учитывающее десятичные значения
    # т.е. если токен имеет 8 десятичных знаков, то для 1 токена значение будет 100000000
    balance: float
    buy_link: Optional[str] = None


# схема токена авторизации
class TokenPayload(BaseModel):
    user_id: int
    balance_check_time: Optional[int] = None  # timestamp последней проверки
    subscription: Optional[enums.SubscriptionPlanType] = None

class AppTokenPayload(TokenPayload):
    exp: int
    iat: int = Field(default_factory=utils_base.now_timestamp)

# бизнес-схемы чата

class MessageGenerationSettings(BaseModel):
    model: Optional[enums.Model] = None
    chat_style: Optional[enums.ChatStyle] = None
    chat_details_level: Optional[enums.ChatDetailsLevel] = None
    
    # Валидаторы, чтобы небыло ошибок при получении старых данных из базы
    # И чтобы не записывать не валидные значения в базу
    @field_validator("model", mode="before")
    def validate_model(cls, v):
        if v is None:
            return None
        try:
            return enums.Model(v)
        except ValueError:
            # Можно логировать здесь
            logger.warning(f"Invalid model: {v}")
            return None

    @field_validator("chat_style", mode="before")
    def validate_chat_style(cls, v):
        if v is None:
            return None
        try:
            return enums.ChatStyle(v)
        except ValueError:
            return None

    @field_validator("chat_details_level", mode="before")
    def validate_chat_details_level(cls, v):
        if v is None:
            return None
        try:
            return enums.ChatDetailsLevel(v)
        except ValueError:
            return None
    
    

class MessageContent(BaseModel):
    message: str
    settings: MessageGenerationSettings

class ToolCallContent(BaseModel):
    name: str
    input: Dict[str, Any]
    output: Dict[str, Any]

# TODO: добавить generation_time_ms
class MessageBase(BaseModel):
    type: enums.MessageType
    content: str
    sender: enums.Role
    recipient: enums.Role
    nonce: int
    created_at: int = Field(default_factory=utils_base.now_timestamp)
    selected_at: int = Field(default_factory=utils_base.now_timestamp)
    generation_time_ms: int = 0
    
    model_config = ConfigDict(from_attributes = True)

class ChatMessage(MessageBase):
    type: Literal[enums.MessageType.TEXT] = enums.MessageType.TEXT
    content: MessageContent

class ToolCallMessage(MessageBase):
    type: Literal[enums.MessageType.TOOL_CALL] = enums.MessageType.TOOL_CALL
    content: ToolCallContent

MessageUnion = Annotated[Union[ChatMessage, ToolCallMessage], Field(discriminator="type")]
MessageUnionAdapter = TypeAdapter(MessageUnion)


class Chat(BaseModel):
    id: int
    title: str
    messages: Dict[int, List[MessageUnion]] # nonce: [message, message, ...]

# апи-схемы чата с ИИ




 
class ChatSummary(BaseModel):
    id: int
    title: str
    created_at: int
    
# бизнес-схемы генерации сообщений

class GenerateMessageSettings(BaseModel):
    model: enums.Model
    chat_style: Optional[enums.ChatStyle] = None
    chat_details_level: Optional[enums.ChatDetailsLevel] = None

class SystemMessage(BaseModel):
    message: str

class AgentFunctionCallingResult(BaseModel):
    success: bool
    new_messages: List[SystemMessage]
    edited_raw_message: str

class FactAboutUser(BaseModel):
    id: int
    description: str
    created_at: int

class UserProfileData(BaseModel):
    preferred_name: Optional[str] = None
    user_context: Optional[str] = None
    facts: List[FactAboutUser]
    addresses: Optional[List[str]] = None
    selected_address: Optional[str] = None
    
class AssistantGenerateData(BaseModel):
    user: UserProfileData
    chat: Chat
    chat_settings: GenerateMessageSettings
    toolbox_ids: List[enums.ToolboxList] = []

    
# Схемы для вызова функций

class FunctionCall(BaseModel):
    name: str
    args: Optional[Any] = None

class AddFactsFunctionCall(FunctionCall):
    name: str = "add_facts"
    args: List[str]
    
class RemoveFactsFunctionCall(FunctionCall):
    name: str = "del_facts"
    args: List[int]
    
class GeneratedResponse(BaseModel):
    text: str
    function_calls: Optional[List[FunctionCall]] = None

from langchain_core.tools import BaseTool, StructuredTool

class Tool(BaseTool):
    pass

class SearchInput(BaseModel):
    query: str = Field(description="Search query")


class MCPServerConfig(BaseModel):
    name: str
    description: str
    transport: str
    
class MCPSSEServer(MCPServerConfig):
    url: str
    transport: str = "sse"
    
class MCPWebSocketServer(MCPServerConfig):
    url: str
    transport: str = "websocket"

class Toolbox(BaseModel):
    id: enums.ToolboxList
    name: str
    description: str
    tools: List[BaseTool]
    type: enums.ToolboxType

class APIErrorContent(BaseModel):
    code: str
    message: str
    details: Optional[Any] = None

class APIErrorResponse(BaseModel):
    error: APIErrorContent


class AIModel(BaseModel):
    id: enums.Model
    provider: enums.Service
    name: str
    description: str
    input_types: list[enums.MessageType] = [enums.MessageType.TEXT, enums.MessageType.TOOL_CALL]
    output_types: list[enums.MessageType] = [enums.MessageType.TEXT, enums.MessageType.TOOL_CALL]

# Схемы для подписок

class SubscriptionPlan(BaseModel):
    id: enums.SubscriptionPlanType
    name: str
    requirements: List[TokenRequirement]
    max_credits: int

# дополненная схема подписки с доп информацие о функциях и моделях
class SubscriptionPlanExtended(SubscriptionPlan):
    available_models_ids: List[enums.Model]
    available_toolboxes_ids: List[enums.ToolboxList]

# cхемы с прописанными ограничениями на план

class AiModelRestricted(AIModel):
    from_plan_id: enums.SubscriptionPlanType

class ToolboxRestricted(Toolbox):
    from_plan_id: enums.SubscriptionPlanType

class PromoCodeActivateRequest(BaseModel):
    code: str

# схемы для Yapps feed

class SourceStatusInProject(BaseModel):
    project_id: int
    source_type: enums.ProjectAccountStatusType

class PostStats(BaseModel):
    favorite_count: int
    retweet_count: int
    reply_count: int
    views_count: Optional[int] = None

class Post(BaseModel):
    text: str
    created_timestamp: int
    full_post_json: Dict[str, Any]
    stats: PostStats
    account_name: str
    # связь между автором поста и проектом
    account_projects_statuses: List[SourceStatusInProject]

class FeedSort(BaseModel):
    type: Optional[enums.SortType] = Field(default=enums.SortType.NEW)

class FeedFilter(BaseModel):
    projects_ids: Optional[List[int]] = None
    include_project_sources: Optional[bool] = Field(default=True)
    include_other_sources: Optional[bool] = Field(default=True)

class GetFeedRequest(BaseModel):
    filter: Optional[FeedFilter] = Field(default=FeedFilter())
    sort: Optional[FeedSort] = Field(default=FeedSort())

class GetFeedResponse(BaseModel):
    posts: List[Post]
    
    
class SelectProjectsRequest(BaseModel):
    project_ids: List[int]
    
    
# ---------- SETTINGS ---------- #
class StyleSettings(BaseModel):
    tonality: Optional[enums.Tonality] = None
    formality: Optional[enums.Formality] = None
    perspective: Optional[enums.Perspective] = None

class StyleSettingsSafe(StyleSettings):
    # Валидаторы для безопасного извлечения из базы данных
    @field_validator("tonality", mode="before")
    def validate_tonality(cls, v):
        if v is None:
            return None
        try:
            return enums.Tonality(v)
        except ValueError:
            logger.warning(f"Invalid tonality: {v}")
            return None

    @field_validator("formality", mode="before")
    def validate_formality(cls, v):
        if v is None:
            return None
        try:
            return enums.Formality(v)
        except ValueError:
            logger.warning(f"Invalid formality: {v}")
            return None

    @field_validator("perspective", mode="before")
    def validate_perspective(cls, v):
        if v is None:
            return None
        try:
            return enums.Perspective(v)
        except ValueError:
            logger.warning(f"Invalid perspective: {v}")
            return None


class ContentSettings(BaseModel):
    length: Optional[enums.LengthOption] = None
    emoji: Optional[enums.EmojiUsage] = None
    hashtags: Optional[enums.HashtagPolicy] = None
    project_mention: Optional[enums.MentionsPolicy] = enums.MentionsPolicy.ANY

class ContentSettingsSafe(ContentSettings):
    # Валидаторы для безопасного извлечения из базы данных
    @field_validator("length", mode="before")
    def validate_length(cls, v):
        if v is None:
            return None
        try:
            return enums.LengthOption(v)
        except ValueError:
            logger.warning(f"Invalid length: {v}")
            return None

    @field_validator("emoji", mode="before")
    def validate_emoji(cls, v):
        if v is None:
            return None
        try:
            return enums.EmojiUsage(v)
        except ValueError:
            logger.warning(f"Invalid emoji: {v}")
            return None

    @field_validator("hashtags", mode="before")
    def validate_hashtags(cls, v):
        if v is None:
            return None
        try:
            return enums.HashtagPolicy(v)
        except ValueError:
            logger.warning(f"Invalid hashtags: {v}")
            return None

    @field_validator("project_mention", mode="before")
    def validate_project_mention(cls, v):
        if v is None:
            return enums.MentionsPolicy.ANY  # Обязательное значение по умолчанию
        try:
            return enums.MentionsPolicy(v)
        except ValueError:
            logger.warning(f"Invalid project_mention: {v}")
            return enums.MentionsPolicy.ANY  # Обязательное значение по умолчанию


class PersonalizationSettings(BaseModel):
    user_social_login: Optional[str] = None
    user_context: Optional[str] = ""
    style: StyleSettings = Field(default_factory=StyleSettings)
    content: ContentSettings = Field(default_factory=ContentSettings)


class PersonalizationSettingsSafe(PersonalizationSettings):
    style: StyleSettingsSafe = Field(default_factory=StyleSettingsSafe)
    content: ContentSettingsSafe = Field(default_factory=ContentSettingsSafe)
    
    @field_validator("user_social_login", mode="before")
    def validate_user_social_login(cls, v):
        if v is None:
            return None
        if not isinstance(v, str):
            logger.warning(f"Invalid user_social_login: {v}")
            return None
        if v == "":
            logger.warning(f"Invalid user_social_login: {v}")
            return None
        return v
    
    @field_validator("user_context", mode="before")
    def validate_user_context(cls, v):
        if v is None:
            return ""
        if not isinstance(v, str):
            logger.warning(f"Invalid user_context: {v}")
            return ""
        return v
    
    # Простые валидаторы, делегирующие работу вложенным классам
    @field_validator("style", mode="before")
    def validate_style(cls, v):
        if v is None:
            return StyleSettingsSafe()
        if isinstance(v, dict):
            try:
                return StyleSettingsSafe.model_validate(v)
            except Exception:
                logger.warning(f"Invalid style settings: {v}")
                return StyleSettingsSafe()
        return v

    @field_validator("content", mode="before")
    def validate_content(cls, v):
        if v is None:
            return ContentSettingsSafe()
        if isinstance(v, dict):
            try:
                return ContentSettingsSafe.model_validate(v)
            except Exception:
                logger.warning(f"Invalid content settings: {v}")
                return ContentSettingsSafe()
        return v

# todo: использовать схему обычного поста или тд

class PostExampleCreate(BaseModel):
    project_id: int
    post_text: str

class PostExample(PostExampleCreate):
    id: int
    created_at: int
    
    model_config = ConfigDict(from_attributes=True)

class AutoYapsGenerationSettings(BaseModel):
    project_feed: List[Post]
    model: AIModel


class LeaderboardUser(BaseModel):
    avatar_url: Optional[str] = None
    name: str
    login: str
    followers: Optional[int] = None
    mindshare: float
    engagement: float
    scores: float
    scores_all_time: float
    posts_period: int
    posts_all_time: int
    is_connected: bool = False
    