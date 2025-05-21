from datetime import datetime, timedelta
from typing import Optional, List, Dict, ClassVar
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Enum as SQLEnum, CheckConstraint
from sqlalchemy.orm import relationship, declarative_base
import time
from sqlalchemy.dialects import postgresql

from src import enums, schemas, utils_base
from src.config.settings import settings

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
    
    # сколько кредитов пользователь использовал сегодня
    used_credits_today = Column(Integer, nullable=False, default=0)
    credits_last_update = Column(Integer, nullable=False, default=utils_base.now_timestamp)
    pro_plan_promo_activated = Column(Boolean, nullable=False, default=False, server_default="false")

    # Relationships
    chats = relationship("Chat", back_populates="user")
    facts = relationship("UserFact", back_populates="user")
    wallet_addresses = relationship("WalletAddress", back_populates="user")
    promo_code_usage = relationship("PromoCodeUsage", back_populates="user")
    
    __table_args__ = (
        CheckConstraint('credits >= 0', name='credits_nonnegative'),
    )
    
    @property
    def subscription_plan(self) -> enums.SubscriptionPlanType:
        if self.pro_plan_promo_activated:
            return enums.SubscriptionPlanType.PRO
        else:
            return enums.SubscriptionPlanType.BASIC

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
    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan")
    
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


class PromoCode(Base):
    __tablename__ = "promo_code"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String, nullable=False, unique=True)
    created_at = Column(Integer, default=utils_base.now_timestamp, nullable=False)
    valid_until = Column(Integer, nullable=False)

    # Relationships
    usage = relationship("PromoCodeUsage", back_populates="promo_code")

class PromoCodeUsage(Base):
    __tablename__ = "promo_code_usage"

    id = Column(Integer, primary_key=True, index=True)
    promo_code_id = Column(Integer, ForeignKey("promo_code.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    used_at = Column(Integer, default=utils_base.now_timestamp, nullable=False)

    # Relationships
    promo_code = relationship("PromoCode", back_populates="usage")
    user = relationship("User", back_populates="promo_code_usage")


# модели для Yapps feed

class Project(Base):
    __tablename__ = "project"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(Integer, default=utils_base.now_timestamp, nullable=False)
    
    name = Column(String, nullable=False)
    description = Column(String, nullable=False)
    url = Column(String, nullable=False)
    
    keywords = Column(String, nullable=False)
    
    # Relationships
    accounts = relationship("ProjectAccountStatus", back_populates="project")
    mentions = relationship("ProjectMention", back_populates="project")
    
class SocialAccount(Base):
    __tablename__ = "social_account"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(Integer, default=utils_base.now_timestamp, nullable=False)
    
    social_id = Column(String, nullable=False) # id аккаунта в соц. сети
    social_login = Column(String, nullable=False) # публичный логин аккаунта в соц. сети
    name = Column(String, nullable=True)
    
    # Relationships
    posts = relationship("SocialPost", back_populates="account")
    projects = relationship("ProjectAccountStatus", back_populates="account")

class SocialPost(Base):
    __tablename__ = "social_post"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(Integer, default=utils_base.now_timestamp, nullable=False)
    
    social_id = Column(String, nullable=False) # id поста в соц. сети
    account_id = Column(Integer, ForeignKey("social_account.id"), nullable=False)
    
    text = Column(String, nullable=False)
    posted_at = Column(Integer, nullable=False)
    
    raw_data = Column(postgresql.JSONB, nullable=True)
    
    # Relationships
    account = relationship("SocialAccount", back_populates="posts")
    statistic = relationship("SocialPostStatistic", back_populates="post")
    mentions = relationship("ProjectMention", back_populates="post")

class SocialPostStatistic(Base):
    __tablename__ = "social_post_statistic"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(Integer, default=utils_base.now_timestamp, nullable=False)
    
    post_id = Column(Integer, ForeignKey("social_post.id"), nullable=False)
    
    likes = Column(Integer, nullable=False)
    comments = Column(Integer, nullable=False)
    reposts = Column(Integer, nullable=False)
    views = Column(Integer, nullable=False)
    
    # Relationships
    post = relationship("SocialPost", back_populates="statistic")
    

class ProjectAccountStatus(Base):
    #Существование этой связи означает, что аккаунт связан с проектом.
    __tablename__ = "project_account_status"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(Integer, default=utils_base.now_timestamp, nullable=False)
    
    project_id = Column(Integer, ForeignKey("project.id"), nullable=False)
    account_id = Column(Integer, ForeignKey("social_account.id"), nullable=False)
    
    type = Column(postgresql.ENUM(enums.ProjectAccountStatusType), nullable=False)
    
    # Relationships
    project = relationship("Project", back_populates="accounts")
    account = relationship("SocialAccount", back_populates="projects")


class ProjectMention(Base):
    #Существование этой связи означает, что пост ссылается на проект.
    __tablename__ = "project_mention"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(Integer, default=utils_base.now_timestamp, nullable=False)
    
    project_id = Column(Integer, ForeignKey("project.id"), nullable=False)
    post_id = Column(Integer, ForeignKey("social_post.id"), nullable=False)

    # Relationships
    project = relationship("Project", back_populates="mentions")
    post = relationship("SocialPost", back_populates="mentions")