from typing import Dict, List, Optional
from decimal import Decimal
from pydantic import BaseModel, Field
from datetime import datetime
import time

from src import enums


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

class UserProfile(BaseModel):
    preferred_name: Optional[str] = None
    user_context: Optional[str] = None

class UserChatSettings(BaseModel):
    preferred_chat_model: Optional[enums.Model] = None
    preferred_chat_style: Optional[enums.ChatStyle] = None
    preferred_chat_details_level: Optional[enums.ChatDetailsLevel] = None



class User(BaseModel):
    address: str
    chain_id: int
    profile: UserProfile
    chat_settings: UserChatSettings



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
    address: str
    chain_id: int
    balance_check_time: int  # timestamp последней проверки
    balance_check_success: bool  # результат проверки

# бизнес-схемы чата

class Message(BaseModel):
    content: str
    sender: enums.Role
    recipient: enums.Role
    model: enums.Model
    nonce: int
    chat_style: Optional[enums.ChatStyle] = None
    chat_details_level: Optional[enums.ChatDetailsLevel] = None
    created_at: int
    selected_at: int

class Chat(BaseModel):
    id: int
    title: str
    messages: Dict[int, List[Message]] # nonce: [message, message, ...]

# апи-схемы чата с ИИ

class MessageCreate(BaseModel):
    chat_id: Optional[int] = None
    nonce: int # для изменения старого сообщения
    message: str
    model: enums.Model
    chat_style: Optional[enums.ChatStyle] = None
    chat_details_level: Optional[enums.ChatDetailsLevel] = None
    
class ChatSummary(BaseModel):
    id: int
    title: str
    last_updated_at: int
    
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
    description: str

class UserProfileData(BaseModel):
    preferred_name: Optional[str] = None
    user_context: Optional[str] = None
    facts: List[FactAboutUser]
    
class AssistantGenerateData(BaseModel):
    user: UserProfileData
    chat: Chat
    chat_settings: GenerateMessageSettings

    
    
