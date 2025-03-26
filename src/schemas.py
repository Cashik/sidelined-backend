from typing import Dict, List, Optional
from decimal import Decimal
from pydantic import BaseModel, Field
from datetime import datetime
import time

from src import enums


# схемы роутера auth

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


class IsLoginResponse(BaseModel):
    logged_in: bool


# схемы роутера user




# внутренние схемы чата с ИИ

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


class MessageCreate(BaseModel):
    chat_id: Optional[int] = None
    nonce: int # для изменения старого сообщения
    message: str
    model: enums.Model
    

class Chat(BaseModel):
    id: int
    title: str
    messages: Dict[int, List[Message]] # nonce: [message, message, ...]

class ChatSummary(BaseModel):
    id: int
    title: str
    last_updated_at: int
    

# схемы роутера chats

    
    
