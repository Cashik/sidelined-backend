from datetime import datetime, timedelta
from typing import Optional, List, Dict, ClassVar
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship, declarative_base
import time

from src import enums, schemas

Base = declarative_base()

# todo: файлы к сообщениям


# Функция для получения текущего timestamp в секундах
def now_timestamp():
    return int(time.time())

# Функция для получения timestamp 50 часов назад
def past_timestamp(hours=50):
    return int(time.time() - hours * 3600)


class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True, index=True)
    address = Column(String)
    chain_id = Column(Integer)
    created_at = Column(Integer, default=now_timestamp)
    
    # Relationships
    chats = relationship("Chat", back_populates="user")

class Chat(Base):
    __tablename__ = "chat"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id"))
    created_at = Column(Integer, default=now_timestamp)
    title = Column(String)
    visible = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User", back_populates="chats")
    messages = relationship("Message", back_populates="chat")
    
class Message(Base):
    __tablename__ = "message"

    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, ForeignKey("chat.id"))
    content = Column(String)
    sender = Column(SQLEnum(enums.Role))
    recipient = Column(SQLEnum(enums.Role))
    model = Column(String)
    nonce = Column(Integer)
    created_at = Column(Integer, default=now_timestamp)
    selected_at = Column(Integer, default=now_timestamp)
    
    # Relationships
    chat = relationship("Chat", back_populates="messages")
    

