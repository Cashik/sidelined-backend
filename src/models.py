from datetime import datetime, timedelta
from typing import Optional, List, Dict, ClassVar
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Enum as SQLEnum, CheckConstraint, Float
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
    personalization_brain_settings = Column(postgresql.JSONB, nullable=True, server_default=None)
    
    twitter_login = Column(String, nullable=True, server_default=None, unique=True)
    
    og_bonus_activated = Column(Boolean, nullable=False, default=False, server_default="false")
    
    # сколько кредитов пользователь использовал сегодня
    used_credits_today = Column(Integer, nullable=False, default=0)
    credits_last_update = Column(Integer, nullable=False, default=utils_base.now_timestamp)
    pro_plan_promo_activated = Column(Boolean, nullable=False, default=False, server_default="false")

    # Relationships
    chats = relationship("Chat", back_populates="user")
    facts = relationship("UserFact", back_populates="user")
    wallet_addresses = relationship("WalletAddress", back_populates="user")
    promo_code_usage = relationship("PromoCodeUsage", back_populates="user")
    selected_projects = relationship("UserSelectedProject", back_populates="user")

    __table_args__ = (
        CheckConstraint('credits >= 0', name='credits_nonnegative'),
    )
    
    def __str__(self) -> str:
        if self.preferred_name:
            return f"User #{self.id} ({self.preferred_name})"
        return f"User #{self.id}"
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, preferred_name='{self.preferred_name}')>"
    
    @property
    def subscription_plan(self) -> enums.SubscriptionPlanType:
        if self.pro_plan_promo_activated:
            return enums.SubscriptionPlanType.PRO
        else:
            return enums.SubscriptionPlanType.BASIC

class WalletAddress(Base):
    __tablename__ = "wallet_address"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    address = Column(String, unique=True, nullable=False)  # Адрес в lowercase
    created_at = Column(Integer, default=utils_base.now_timestamp, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="wallet_addresses")

class UserFact(Base):
    __tablename__ = "user_fact"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    description = Column(String(200), nullable=False)
    created_at = Column(Integer, default=utils_base.now_timestamp, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="facts")
    

class Chat(Base):
    __tablename__ = "chat"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
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
    chat_id = Column(Integer, ForeignKey("chat.id", ondelete="CASCADE"), nullable=False)
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
    promo_code_id = Column(Integer, ForeignKey("promo_code.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
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
    description = Column(String, nullable=True)
    url = Column(String, nullable=True)
    
    icon_url = Column(String, nullable=False, default="", server_default="")
    keywords = Column(String, nullable=False)
    
    search_min_likes = Column(Integer, nullable=True, default=None)
    
    is_selected_by_default = Column(Boolean, nullable=False, default=False, server_default="false")
    is_leaderboard_project = Column(Boolean, nullable=False, default=False, server_default="false")
    
    # Relationships
    accounts = relationship("ProjectAccountStatus", back_populates="project", cascade="all, delete-orphan")
    mentions = relationship("ProjectMention", back_populates="project", cascade="all, delete-orphan")
    user_selected_projects = relationship("UserSelectedProject", back_populates="project", cascade="all, delete-orphan")
    posts_templates = relationship("PostTemplate", back_populates="project", cascade="all, delete-orphan")
    leaderboard_history = relationship("ProjectLeaderboardHistory", back_populates="project", cascade="all, delete-orphan")

    def __str__(self) -> str:
        return f"{self.name}"
    
    def __repr__(self) -> str:
        return f"<Project(id={self.id}, name='{self.name}')>"
    
    def __admin_repr__(self, request) -> str:
        """Display representation in admin"""
        return self.name
    
    def __admin_select2_repr__(self, request) -> str:
        """Display representation in select2 dropdowns"""
        return f'<span><strong>{self.name}</strong></span>'
    

class UserSelectedProject(Base):
    __tablename__ = "user_selected_project"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    project_id = Column(Integer, ForeignKey("project.id", ondelete="CASCADE"), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="selected_projects")
    project = relationship("Project", back_populates="user_selected_projects")


class SocialAccount(Base):
    __tablename__ = "social_account"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(Integer, default=utils_base.now_timestamp, nullable=False)
    
    social_id = Column(String, nullable=False) # id аккаунта в соц. сети
    social_login = Column(String, nullable=False) # публичный логин аккаунта в соц. сети
    name = Column(String, nullable=True)
    
    twitter_scout_score = Column(Float, nullable=True)
    twitter_scout_score_updated_at = Column(Integer, nullable=True)
    
    # последние данные которые мы извлекли при парсинге
    last_avatar_url = Column(String, nullable=True, default=None, server_default=None)
    last_followers_count = Column(Integer, nullable=True, default=None, server_default=None)
    
    # флаг, чтобы не учитывать аккаунт в лидерборде
    is_disabled_for_leaderboard = Column(Boolean, nullable=False, default=False, server_default="false")
    
    # Relationships
    posts = relationship("SocialPost", back_populates="account", cascade="all, delete-orphan")
    projects = relationship("ProjectAccountStatus", back_populates="account")
    scores = relationship(
        "ScorePayout",
        back_populates="social_account",
        cascade="all, delete-orphan"
    )
    
    def __str__(self) -> str:
        if self.name:
            return f"{self.social_login} ({self.name})"
        return f"{self.social_login}"
    
    def __repr__(self) -> str:
        return f"<SocialAccount(id={self.id}, social_login='{self.social_login}')>"
    
    def __admin_repr__(self, request) -> str:
        """Display representation in admin"""
        if self.name:
            return f"{self.social_login} ({self.name})"
        return self.social_login
    
    def __admin_select2_repr__(self, request) -> str:
        """Display representation in select2 dropdowns"""
        if self.name:
            return f'<span><strong>{self.social_login}</strong> <small>({self.name})</small></span>'
        return f'<span><strong>{self.social_login}</strong></span>'
    

class SocialPost(Base):
    __tablename__ = "social_post"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(Integer, default=utils_base.now_timestamp, nullable=False)
    
    social_id = Column(String, nullable=False) # id поста в соц. сети
    account_id = Column(Integer, ForeignKey("social_account.id", ondelete="CASCADE"), nullable=False)
    
    text = Column(String, nullable=False)
    posted_at = Column(Integer, nullable=False)
    
    raw_data = Column(postgresql.JSONB, nullable=True)
    
    # Relationships
    account = relationship("SocialAccount", back_populates="posts")
    statistic = relationship("SocialPostStatistic", back_populates="post", cascade="all, delete-orphan")
    mentions = relationship("ProjectMention", back_populates="post", cascade="all, delete-orphan")


class SocialPostStatistic(Base):
    __tablename__ = "social_post_statistic"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(Integer, default=utils_base.now_timestamp, nullable=False)
    
    post_id = Column(Integer, ForeignKey("social_post.id", ondelete="CASCADE"), nullable=False)
    
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
    
    project_id = Column(Integer, ForeignKey("project.id", ondelete="CASCADE"), nullable=False)
    account_id = Column(Integer, ForeignKey("social_account.id", ondelete="CASCADE"), nullable=False)
    
    type = Column(postgresql.ENUM(enums.ProjectAccountStatusType), nullable=False)
    
    # Relationships
    project = relationship("Project", back_populates="accounts")
    account = relationship("SocialAccount", back_populates="projects")
    
    def __str__(self) -> str:
        try:
            # Попытка получить имена через relationships
            project_name = self.project.name if self.project else f"Project#{self.project_id}"
            account_login = self.account.social_login if self.account else f"Account#{self.account_id}"
            return f"{project_name} ↔ {account_login} ({self.type.value})"
        except Exception:
            # Fallback на простое отображение без relationships
            return f"Project#{self.project_id} ↔ Account#{self.account_id} ({self.type.value})"
    
    def __repr__(self) -> str:
        return f"<ProjectAccountStatus(id={self.id}, project_id={self.project_id}, account_id={self.account_id}, type={self.type})>"
    
    def __admin_repr__(self, request) -> str:
        """Display representation in admin"""
        try:
            # Попытка получить имена через relationships
            project_name = self.project.name if self.project else f"Project#{self.project_id}"
            account_login = self.account.social_login if self.account else f"Account#{self.account_id}"
            return f"{project_name} ↔ {account_login} ({self.type.value})"
        except Exception:
            # Fallback на простое отображение без relationships
            return f"Project#{self.project_id} ↔ Account#{self.account_id} ({self.type.value})"
    
    def __admin_select2_repr__(self, request) -> str:
        """Display representation in select2 dropdowns"""
        try:
            project_name = self.project.name if self.project else f"Project#{self.project_id}"
            account_login = self.account.social_login if self.account else f"Account#{self.account_id}"
            return f'<span><strong>{project_name}</strong> ↔ <strong>{account_login}</strong> <small>({self.type.value})</small></span>'
        except Exception:
            return f'<span>Project#{self.project_id} ↔ Account#{self.account_id} <small>({self.type.value})</small></span>'


class ProjectMention(Base):
    #Существование этой связи означает, что пост ссылается на проект.
    __tablename__ = "project_mention"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(Integer, default=utils_base.now_timestamp, nullable=False)
    
    project_id = Column(Integer, ForeignKey("project.id", ondelete="CASCADE"), nullable=False)
    post_id = Column(Integer, ForeignKey("social_post.id", ondelete="CASCADE"), nullable=False)

    # Relationships
    project = relationship("Project", back_populates="mentions")
    post = relationship("SocialPost", back_populates="mentions")


class AdminUser(Base):
    """Администратор/модератор системы"""
    __tablename__ = "admin_user"

    id = Column(Integer, primary_key=True, index=True)
    login = Column(String(50), unique=True, nullable=False, index=True)
    password_hash = Column(String(128), nullable=False)
    role = Column(postgresql.ENUM(enums.AdminRole, name="admin_role"), nullable=False, server_default=enums.AdminRole.ADMIN.value)
    created_at = Column(Integer, default=utils_base.now_timestamp, nullable=False)
    last_login_at = Column(Integer, nullable=True)

    def verify_password(self, password: str) -> bool:
        """Проверка пароля при помощи bcrypt"""
        try:
            import bcrypt  # локальный импорт, чтобы не требовать bcrypt, если метод не вызывают
            return bcrypt.checkpw(password.encode(), self.password_hash.encode())
        except Exception:
            return False
    
    def __str__(self) -> str:
        return f"{self.login} ({self.role.value})"
    
    def __repr__(self) -> str:
        return f"<AdminUser(id={self.id}, login='{self.login}', role={self.role})>"
    
    def __admin_repr__(self, request) -> str:
        return f"{self.login} ({self.role.value})"


class PostTemplate(Base):
    __tablename__ = "post_template"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(Integer, default=utils_base.now_timestamp, nullable=False)
    
    project_id = Column(Integer, ForeignKey("project.id", ondelete="CASCADE"), nullable=False)
    post_text = Column(String, nullable=False)
    
    # Relationships
    project = relationship("Project", back_populates="posts_templates")
    
    def __str__(self) -> str:
        return f"PostTemplate #{self.id} for project {self.project_id}"
    
    def __repr__(self) -> str:
        return f"<PostTemplate(id={self.id}, project_id={self.project_id})>"


class ProjectLeaderboardHistory(Base):
    # история лидерборда проекта
    
    __tablename__ = "project_leaderboard_history"
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(Integer, default=utils_base.now_timestamp, nullable=False)
    
    start_ts = Column(Integer, nullable=False)
    end_ts = Column(Integer, nullable=False)
    
    project_id = Column(Integer, ForeignKey("project.id", ondelete="CASCADE"), nullable=False)
    
    # Relationships
    project = relationship("Project", back_populates="leaderboard_history")
    scores = relationship("ScorePayout", back_populates="project_leaderboard_history", cascade="all, delete-orphan")


class ScorePayout(Base):
    __tablename__ = "score_payout"
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(Integer, default=utils_base.now_timestamp, nullable=False)
    
    # проект, за который выплачивается score
    project_id = Column(Integer, ForeignKey("project.id", ondelete="CASCADE"), nullable=False)
    # аккаунт, которому выплачивается score
    social_account_id = Column(Integer, ForeignKey("social_account.id", ondelete="CASCADE"), nullable=False)
    # id истории лидерборда, если есть
    project_leaderboard_history_id = Column(Integer, ForeignKey("project_leaderboard_history.id", ondelete="SET NULL"), nullable=True)
    
    # финальный score, который выплачивается
    score = Column(Float, nullable=False)
    
    # статистические данные для лидерборда и деталей выплаты
    engagement = Column(Float, nullable=False) # engagement, который использовался для расчета mindshare
    mindshare = Column(Float, nullable=False) # mindshare, который использовался для расчета score
    base_score = Column(Float, nullable=False) # базовый рейтинг, на основе текущего mindshare
    
    # TODO: добавить колонки для расчета бонусов
    new_posts_count = Column(Integer, nullable=True, default=None, server_default=None)
    # дата первого поста об проекте лидерборда с момента привязки юзером 
    first_post_at = Column(Integer, nullable=True, default=None, server_default=None)
    # дата последнего упоминания об проекте лидерборда с момента привязки юзером
    last_post_at = Column(Integer, nullable=True, default=None, server_default=None)
    # дата начала последнего недельного стрика
    weekly_streak_start_at = Column(Integer, nullable=True, default=None, server_default=None)
    # очки лояльности. Даются за присутствие в майндшере. Отнимаются за неактивность.
    loyalty_points = Column(Float, nullable=True, default=None, server_default=None)
    # отметка лояльности ниже которой нельзя опуститься. Повышается при достижении некоторых значений.
    min_loyalty = Column(Integer, nullable=True, default=None, server_default=None)
    
    # Relationships
    social_account = relationship("SocialAccount", back_populates="scores")
    project_leaderboard_history = relationship('ProjectLeaderboardHistory', back_populates='scores')
    


    