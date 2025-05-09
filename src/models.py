from datetime import datetime, timedelta
from typing import Optional, List, Dict, ClassVar
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Enum as SQLEnum, CheckConstraint
from sqlalchemy.orm import relationship, declarative_base
import time
from sqlalchemy.dialects import postgresql

from src import enums, schemas, utils_base
from src.config import settings

Base = declarative_base()

# !Внимание: не забывать про nullable=False при создании новых полей

class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(Integer, default=utils_base.now_timestamp, nullable=False)
    
    # информация о пользователе
    preferred_name = Column(String(20), nullable=True)
    user_context = Column(String(500), nullable=True)
    chat_settings = Column(postgresql.JSONB, nullable=True, server_default=None)
    
    credits = Column(Integer, nullable=False, default=settings.DEFAULT_CREDITS) #!Warning: менять значение только с помощью методов из crud.py
    credits_last_update = Column(Integer, nullable=False, default=utils_base.now_timestamp)

    # Relationships
    chats = relationship("Chat", back_populates="user")
    facts = relationship("UserFact", back_populates="user")
    wallet_addresses = relationship("WalletAddress", back_populates="user")
    __table_args__ = (
        CheckConstraint('credits >= 0', name='credits_nonnegative'),
    )

class WalletAddress(Base):
    __tablename__ = "wallet_address"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    address = Column(String, unique=True, nullable=False)  # Адрес в lowercase
    created_at = Column(Integer, default=utils_base.now_timestamp, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="wallet_addresses")

class UserFact(Base):
    __tablename__ = "user_fact"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    description = Column(String(200), nullable=False)
    created_at = Column(Integer, default=utils_base.now_timestamp, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="facts")
    

class Chat(Base):
    __tablename__ = "chat"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    created_at = Column(Integer, default=utils_base.now_timestamp, nullable=False)
    title = Column(String, nullable=False)
    visible = Column(Boolean, default=True, nullable=False)
    last_analysed_nonce = Column(Integer, default=-1, nullable=False)  # -1 означает, что сообщения еще не анализировались
    
    # Relationships
    user = relationship("User", back_populates="chats")
    messages = relationship("Message", back_populates="chat")
    
class Message(Base):
    __tablename__ = "message"

    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, ForeignKey("chat.id"), nullable=False)
    type = Column(postgresql.ENUM(enums.MessageType), nullable=False)
    content = Column(postgresql.JSONB, nullable=True)
    sender = Column(postgresql.ENUM(enums.Role), nullable=False)
    recipient = Column(postgresql.ENUM(enums.Role), nullable=False)
    nonce = Column(Integer, nullable=False)
    created_at = Column(Integer, default=utils_base.now_timestamp, nullable=False)
    selected_at = Column(Integer, default=utils_base.now_timestamp, nullable=False)
    generation_time_ms = Column(Integer, default=0, nullable=False)
    
    # Relationships
    chat = relationship("Chat", back_populates="messages")
    

