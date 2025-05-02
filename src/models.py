from datetime import datetime, timedelta
from typing import Optional, List, Dict, ClassVar
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship, declarative_base
import time
from sqlalchemy.dialects import postgresql

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
    created_at = Column(Integer, default=now_timestamp)
    
    # информация о пользователе
    preferred_name = Column(String(20), nullable=True)
    user_context = Column(String(500), nullable=True)
    preferred_chat_model = Column(postgresql.ENUM(enums.Model), nullable=True, default=None)
    preferred_chat_style = Column(postgresql.ENUM(enums.ChatStyle), nullable=True, default=None)
    preferred_chat_details_level = Column(postgresql.ENUM(enums.ChatDetailsLevel), nullable=True, default=None)
    
    # Relationships
    chats = relationship("Chat", back_populates="user")
    facts = relationship("UserFact", back_populates="user")
    wallet_addresses = relationship("WalletAddress", back_populates="user")

class WalletAddress(Base):
    __tablename__ = "wallet_address"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id"))
    address = Column(String, unique=True)  # Адрес в lowercase
    created_at = Column(Integer, default=now_timestamp)
    
    # Relationships
    user = relationship("User", back_populates="wallet_addresses")

class UserFact(Base):
    __tablename__ = "user_fact"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id"))
    description = Column(String(200))
    created_at = Column(Integer, default=now_timestamp)
    
    # Relationships
    user = relationship("User", back_populates="facts")
    

class Chat(Base):
    __tablename__ = "chat"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id"))
    created_at = Column(Integer, default=now_timestamp)
    title = Column(String)
    visible = Column(Boolean, default=True)
    last_analysed_nonce = Column(Integer, default=-1)  # -1 означает, что сообщения еще не анализировались
    
    # Relationships
    user = relationship("User", back_populates="chats")
    messages = relationship("Message", back_populates="chat")
    
class Message(Base):
    __tablename__ = "message"

    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, ForeignKey("chat.id"))
    type = Column(postgresql.ENUM(enums.MessageType))
    content = Column(postgresql.JSONB)
    sender = Column(postgresql.ENUM(enums.Role))
    recipient = Column(postgresql.ENUM(enums.Role))
    nonce = Column(Integer)
    created_at = Column(Integer, default=now_timestamp)
    selected_at = Column(Integer, default=now_timestamp)
    generation_time_ms = Column(Integer, default=0)
    
    # Relationships
    chat = relationship("Chat", back_populates="messages")
    

