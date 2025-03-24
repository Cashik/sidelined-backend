from datetime import datetime, timedelta
from typing import Optional, List, Dict, ClassVar
from sqlmodel import Field, SQLModel, Enum, Relationship

import time

from src import enums, schemas

# todo: файлы к сообщениям


# Функция для получения текущего timestamp в секундах
def now_timestamp():
    return int(time.time())

# Функция для получения timestamp 50 часов назад
def past_timestamp(hours=50):
    return int(time.time() - hours * 3600)


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    address: str
    chain_id: int
    created_at: int = Field(default_factory=now_timestamp)
    
    # Relationships
    chats: List["Chat"] = Relationship(back_populates="user")

class Chat(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    created_at: int = Field(default_factory=now_timestamp)
    title: str
    visible: bool = Field(default=True)
    
    # Relationships
    user: User = Relationship(back_populates="chats")
    messages: List["Message"] = Relationship(back_populates="chat")
    
class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    chat_id: int = Field(foreign_key="chat.id")
    content: str
    # поля для определения роли и получателей сообщения
    sender: enums.Role = Field(sa_column=Enum(enums.Role))
    recipient: enums.Role = Field(sa_column=Enum(enums.Role))
    # источник генерации (последний выбранный)
    service: enums.Service = Field(sa_column=Enum(enums.Service))
    model: enums.Model = Field(sa_column=Enum(enums.Model))
    # поля для определения порядка сообщений
    nonce: int # порядковый номер сообщения в чате
    created_at: int = Field(default_factory=now_timestamp)
    selected_at: int = Field(default_factory=now_timestamp) # для выбора одного из нескольких вариантов с одинаковым порядковым номером
    
    # Relationships
    chat: Chat = Relationship(back_populates="messages")
    

