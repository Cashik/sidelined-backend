from typing import Annotated, Any, Dict, List, Literal, Optional, Union
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
    chat_available: bool


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
    credits: int
    chat_settings: UserChatSettings
    connected_wallets: List[WalletAddress]


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
    
# схема токена авторизации

class TokenPayload(BaseModel):
    user_id: int
    balance_check_time: int  # timestamp последней проверки
    balance_check_success: bool  # результат проверки

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


from mcp.types import Tool, ListToolsResult

class MCPServer(BaseModel):
    name: str
    description: str
    transport: str
    
class MCPSSEServer(MCPServer):
    url: str
    transport: str = "sse"
    
class MCPWebSocketServer(MCPServer):
    url: str
    transport: str = "websocket"

class Toolbox(BaseModel):
    name: str
    description: str
    tools: List[Tool]

class APIErrorContent(BaseModel):
    code: str
    message: str
    details: Optional[Any] = None

class APIErrorResponse(BaseModel):
    error: APIErrorContent

