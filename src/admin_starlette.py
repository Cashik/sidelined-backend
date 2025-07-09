from __future__ import annotations

"""Starlette-Admin initialisation.

This module provides a drop-in replacement for the previous SQLAdmin-based
implementation (see ``src.admin``).  Only basic CRUD functionality and the
existing role model (ADMIN / MODERATOR) are migrated for now; advanced
formatters and AJAX widgets will be ported later.

Usage in ``main.py``:

    from src.admin_starlette import setup_admin
    setup_admin(app)

The public API is intentionally kept identical (``setup_admin``) to simplify
switching back if needed.
"""

import logging
from typing import Any, List

from starlette.requests import Request
from starlette_admin.auth import AuthProvider, AdminUser, AdminConfig, LoginFailed
from starlette_admin.contrib.sqla import Admin as SAAdmin, ModelView
from starlette_admin.fields import StringField, IntegerField, DateTimeField, BooleanField, JSONField, TextAreaField, HasOne, HasMany, EnumField, PasswordField, FloatField
from starlette_admin.exceptions import FormValidationError
from starlette.responses import Response

from sqlalchemy.orm import Session
import bcrypt

from src.database import engine, SessionLocal
from src.models import (
    AdminUser as AdminUserModel,
    User,
    Project,
    SocialAccount,
    ProjectAccountStatus,
    ProjectLeaderboardHistory,
    ScorePayout,
)
from src import enums

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers & permissions
# ---------------------------------------------------------------------------

def _has_full_access(request: Request) -> bool:
    return request.session.get("admin_role") == enums.AdminRole.ADMIN.value


# ---------------------------------------------------------------------------
# Authentication Provider
# ---------------------------------------------------------------------------

class BasicAuthProvider(AuthProvider):
    """Username / password authentication backed by ``admin_user`` table."""

    async def login(
        self,
        username: str,  # type: ignore[override]
        password: str,  # type: ignore[override]
        remember_me: bool,  # noqa: FBT001  (not used but required by signature)
        request: Request,
        response: Response,
    ) -> Response:
        if not username or not password:
            raise LoginFailed("Username and password are required")

        db: Session = SessionLocal()
        try:
            admin: AdminUserModel | None = (
                db.query(AdminUserModel).filter(AdminUserModel.login == username).first()
            )
            if admin is None or not admin.verify_password(password):
                logger.warning("Admin login failed for user '%s'", username)
                raise LoginFailed("Invalid credentials")

            # success – remember in session
            request.session.update(
                {
                    "admin_id": admin.id,
                    "admin_login": admin.login,
                    "admin_role": admin.role.value,
                }
            )
            admin.last_login_at = int(__import__("time").time())
            db.commit()
            return response
        finally:
            db.close()

    async def is_authenticated(self, request: Request) -> bool:  # type: ignore[override]
        if "admin_id" in request.session:
            # make user object available downstream if needed
            request.state.admin_user_id = request.session["admin_id"]
            request.state.admin_role = request.session.get("admin_role")
            return True
        return False

    # Optional – show logged-in user in the header
    def get_admin_user(self, request: Request) -> AdminUser | None:  # noqa: D401
        admin_login = request.session.get("admin_login")
        if not admin_login:
            return None
        return AdminUser(username=admin_login)

    # Optional – customise title per user role
    def get_admin_config(self, request: Request) -> AdminConfig | None:  # noqa: D401
        role = request.session.get("admin_role", "")
        return AdminConfig(app_title=f"Sidelined Admin ({role})")

    async def logout(self, request: Request, response: Response) -> Response:  # type: ignore[override]
        request.session.clear()
        return response


# ---------------------------------------------------------------------------
# Base views with RBAC
# ---------------------------------------------------------------------------

class BaseProtectedView(ModelView):
    """Base view - accessible to both ADMIN and MODERATOR."""

    # Visibility in side-menu
    def is_accessible(self, request: Request) -> bool:  # type: ignore[override]
        return "admin_role" in request.session

    def is_visible(self, request: Request) -> bool:  # type: ignore[override]
        return self.is_accessible(request)

    # CRUD permissions - MODERATOR has full access to most models
    def can_create(self, request: Request) -> bool:  # type: ignore[override]
        return "admin_role" in request.session

    def can_edit(self, request: Request) -> bool:  # type: ignore[override]
        return "admin_role" in request.session

    def can_delete(self, request: Request) -> bool:  # type: ignore[override]
        return "admin_role" in request.session


class AdminOnlyView(BaseProtectedView):
    """Visible & editable only for ADMIN - for sensitive models like AdminUser and User."""

    def is_accessible(self, request: Request) -> bool:  # type: ignore[override]
        return _has_full_access(request)

    def is_visible(self, request: Request) -> bool:  # type: ignore[override]
        return self.is_accessible(request)

    def can_create(self, request: Request) -> bool:  # type: ignore[override]
        return _has_full_access(request)

    def can_edit(self, request: Request) -> bool:  # type: ignore[override]
        return _has_full_access(request)

    def can_delete(self, request: Request) -> bool:  # type: ignore[override]
        return _has_full_access(request)


class AdminWriteModeratorReadView(BaseProtectedView):
    """
    Base view for models that are read-only for MODERATOR
    and fully accessible for ADMIN.
    """

    # is_accessible and is_visible are fine from BaseProtectedView
    # (visible to both roles)

    def can_create(self, request: Request) -> bool:  # type: ignore[override]
        return _has_full_access(request)

    def can_edit(self, request: Request) -> bool:  # type: ignore[override]
        return _has_full_access(request)

    def can_delete(self, request: Request) -> bool:  # type: ignore[override]
        return _has_full_access(request)


# ---------------------------------------------------------------------------
# Concrete model views
# ---------------------------------------------------------------------------

# The following views are intentionally minimal.  Advanced customisations
# (search, filters, field ordering, custom formatters) will be re-implemented
# in a later iteration.


class AdminUserAdmin(AdminOnlyView):
    page_size = 100
    searchable_fields = ["login"]
    sortable_fields = ["id", "login", "role", "created_at", "last_login_at"]
    
    fields = [
        IntegerField("id", label="ID", read_only=True),
        StringField("login", label="Login", required=True, help_text="Unique administrator login"),
        PasswordField("password", label="Password", required=True, help_text="Password for login (will be hashed)"),
        PasswordField("new_password", label="New Password", required=False, help_text="Leave empty if you don't want to change password"),
        EnumField("role", label="Role", enum=enums.AdminRole, required=True, help_text="Administrator role"),
        IntegerField("created_at", label="Created At", read_only=True),
        IntegerField("last_login_at", label="Last Login (timestamp)", read_only=True),
    ]
    
    # Исключаем password_hash из всех представлений (он не должен быть виден)
    exclude_fields_from_list = ["password_hash", "new_password"]
    exclude_fields_from_detail = ["password_hash", "password", "new_password"]
    exclude_fields_from_edit = ["password_hash", "password"]  # В редактировании показываем только new_password
    exclude_fields_from_create = ["id", "created_at", "last_login_at", "password_hash", "new_password"]  # Системные поля
    
    async def before_create(self, request: Request, data: dict, obj: Any) -> None:
        """Хеширование пароля перед созданием администратора"""
        # Проверяем уникальность логина
        if "login" in data:
            login = data["login"]
            db: Session = SessionLocal()
            try:
                existing = db.query(AdminUserModel).filter(AdminUserModel.login == login).first()
                if existing:
                    raise FormValidationError({"login": f"Administrator with login '{login}' already exists"})
            finally:
                db.close()
        
        # Хешируем пароль
        if "password" in data:
            password = data.pop("password")  # Убираем пароль из данных
            if not password:
                raise FormValidationError({"password": "Password cannot be empty"})
            
            if len(password) < 6:
                raise FormValidationError({"password": "Password must contain at least 6 characters"})
            
            # Хешируем пароль
            password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
            obj.password_hash = password_hash
            logger.info(f"Создание нового администратора: {data.get('login', 'unknown')}")
    
    async def before_edit(self, request: Request, data: dict, obj: Any) -> None:
        """Обработка смены пароля при редактировании администратора"""
        # Проверяем уникальность логина (если он изменился)
        if "login" in data and data["login"] != obj.login:
            login = data["login"]
            db: Session = SessionLocal()
            try:
                existing = db.query(AdminUserModel).filter(AdminUserModel.login == login).first()
                if existing and existing.id != obj.id:
                    raise FormValidationError({"login": f"Administrator with login '{login}' already exists"})
            finally:
                db.close()
        
        # Обрабатываем смену пароля
        if "new_password" in data:
            new_password = data.pop("new_password")  # Убираем из данных
            if new_password:  # Если пароль указан
                if len(new_password) < 6:
                    raise FormValidationError({"new_password": "Password must contain at least 6 characters"})
                
                # Хешируем новый пароль
                password_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
                obj.password_hash = password_hash
                logger.info(f"Пароль изменен для администратора: {obj.login}")
        
        logger.info(f"Редактирование администратора: {obj.login}")


class UserAdmin(AdminWriteModeratorReadView):
    page_size = 100
    searchable_fields = ["preferred_name", "twitter_login"]
    sortable_fields = ["id", "created_at", "preferred_name", "twitter_login"]
    exclude_fields_from_list = ["chat_settings", "user_context"]
    
    fields = [
        IntegerField("id", label="ID", read_only=True),
        StringField("preferred_name", label="Preferred Name", required=False, help_text="User's preferred name"),
        StringField("twitter_login", label="Twitter Login", help_text="User's Twitter login"),
        TextAreaField("user_context", label="User Context", required=False, help_text="User context information"),
        IntegerField("used_credits_today", label="Credits Used Today", help_text="Credits used today"),
        BooleanField("pro_plan_promo_activated", label="Pro Plan Active", help_text="Is Pro plan promo activated"),
        JSONField("chat_settings", label="Chat Settings", required=False, help_text="Chat settings (JSON)"),
        IntegerField("created_at", label="Created At", read_only=True),
    ]
    
    # Different fields for create and edit forms
    exclude_fields_from_create = ["id", "created_at", "used_credits_today"]  # Don't set used credits when creating
    exclude_fields_from_edit = ["id", "created_at"]  # Can edit credits when updating


class ProjectAdmin(BaseProtectedView):
    page_size = 100
    searchable_fields = ["name", "description", "keywords"]
    sortable_fields = ["id", "name", "created_at"]
    exclude_fields_from_list = ["description", "keywords"]
    
    fields = [
        IntegerField("id", label="ID", read_only=True),
        StringField("name", label="Project Name", required=True, help_text="Name of the project"),
        TextAreaField("description", label="Description", required=False, help_text="Project description"),
        StringField("url", label="Project URL", required=False, help_text="Official project website"),
        StringField("icon_url", label="Icon URL", required=False, help_text="Icon URL of the project (can be taken from the coingecko.com or oficial site)."),
        TextAreaField("keywords", label="Keywords", required=True, help_text="Search keywords (; separated). Example: 'keyword1;key word2;@keyword3'"),
        IntegerField("search_min_likes", label="Min likes for search", required=False, help_text="Leave empty to use the default value for this project."),
        BooleanField("is_leaderboard_project", label="Leaderboard enabled", help_text="Turn on to enable leaderboard for this project. MUST BE ONLY ONE LEADERBOARD PROJECT."),
        BooleanField("is_selected_by_default", label="Selected by default for new users", help_text="Add this project to new users automatically."),
        HasMany("accounts", label="Related Accounts", identity="project-account-status"),
        IntegerField("created_at", label="Created At", read_only=True),
    ]
    
    # Exclude system fields from forms
    exclude_fields_from_create = ["id", "created_at", "accounts"]
    exclude_fields_from_edit = ["id", "created_at", "accounts"]

    async def before_edit(self, request: Request, data: dict, obj: Any) -> None:
        # Только админ может менять is_leaderboard_project
        role = request.session.get("admin_role")
        if (
            "is_leaderboard_project" in data
            and getattr(obj, "is_leaderboard_project", None) != data["is_leaderboard_project"]
            and role != enums.AdminRole.ADMIN.value
        ):
            raise FormValidationError({"is_leaderboard_project": "Только администратор может изменять это поле."})
        # Обработка пустых значений для новых полей
        if "search_min_likes" in data and (data["search_min_likes"] == "" or data["search_min_likes"] is None):
            data["search_min_likes"] = None
        
    async def before_create(self, request: Request, data: dict, obj: Any) -> None:
        # Только админ может выставлять is_leaderboard_project при создании
        role = request.session.get("admin_role")
        if (
            "is_leaderboard_project" in data
            and data["is_leaderboard_project"]
            and role != enums.AdminRole.ADMIN.value
        ):
            raise FormValidationError({"is_leaderboard_project": "Только администратор может устанавливать это поле."})
        # Обработка пустых значений для новых полей
        if "search_min_likes" in data and (data["search_min_likes"] == "" or data["search_min_likes"] is None):
            data["search_min_likes"] = None


class SocialAccountAdmin(BaseProtectedView):
    page_size = 100
    searchable_fields = ["social_login", "name"]
    sortable_fields = ["id", "social_login", "created_at", "twitter_scout_score", "twitter_scout_score_updated_at"]
    
    # Fields for list and detail view
    fields = [
        IntegerField("id", label="ID", read_only=True),
        StringField("social_id", label="Social ID", read_only=True, help_text="Unique account ID in social network (auto-generated)"),
        StringField("social_login", label="Social Login", required=True, help_text="Public login/username"),
        StringField("name", label="Display Name", required=False, help_text="Display name of the user"),
        BooleanField("is_disabled_for_leaderboard", label="Disable for Leaderboard", help_text="Enable this option to exclude the account from leaderboard and hide it from users."),
        HasMany("projects", label="Related Projects", identity="project-account-status"),
        IntegerField("created_at", label="Created At", read_only=True),
        FloatField("twitter_scout_score", label="Twitter Score", required=False),
        IntegerField("twitter_scout_score_updated_at", label="Twitter Score Updated At (timestamp)", required=False),
    ]
    
    # Fields excluded from create form - social_id will be auto-generated
    exclude_fields_from_create = ["id", "created_at", "projects", "social_id"]
    
    # Fields excluded from edit form - only name can be edited
    exclude_fields_from_edit = ["id", "created_at", "social_id", "social_login", "projects"]
    

    
    async def _populate_obj(
        self,
        request: Request,
        obj: Any,
        data: dict,
        is_edit: bool = False,
    ) -> Any:
        """Override to generate social_id before populating object"""
        # For create operations, generate social_id
        if not is_edit and "social_login" in data and not hasattr(obj, 'social_id') or not getattr(obj, 'social_id', None):
            social_login = str(data["social_login"]).strip()
            if len(social_login) < 1:
                raise FormValidationError({"social_login": "Social Login cannot be empty"})
            
            # Get social_id from X API service
            try:
                from src.services.x_api_service import XApiService
                x_service = XApiService()
                social_id = await x_service.get_user_social_id(social_login)
                
                if not social_id:
                    raise FormValidationError({"social_login": "Failed to get social ID from X API service"})
                
                # Set social_id directly on the object
                obj.social_id = social_id
                logger.info(f"Generated social_id '{social_id}' for user '{social_login}'")
                
            except Exception as e:
                logger.error(f"Error getting social_id: {e}")
                raise FormValidationError({"social_login": f"Error getting social ID: {str(e)}"})
        
        # Call parent method
        return await super()._populate_obj(request, obj, data, is_edit)
    
    async def before_edit(self, request: Request, data: dict, obj: Any) -> None:
        """Validation before editing social account - only name can be changed"""
        # When editing, only name can be changed, no additional validation needed
        # social_login and social_id are read-only in edit form
        pass


class ProjectAccountStatusAdmin(BaseProtectedView):
    page_size = 100
    searchable_fields = ["project.name", "account.social_login"]
    sortable_fields = ["id", "created_at", "type"]
    
    fields = [
        IntegerField("id", label="ID", read_only=True),
        HasOne("project", label="Project", required=True, identity="project"),
        HasOne("account", label="Account", required=True, identity="social-account"),
        EnumField("type", label="Status Type", enum=enums.ProjectAccountStatusType, required=True, help_text="Type of relationship between project and account"),
        IntegerField("created_at", label="Created At", read_only=True),
    ]
    
    # Exclude system fields from forms
    exclude_fields_from_create = ["id", "created_at"]
    exclude_fields_from_edit = ["id", "created_at"]
    
    def get_list_query(self, request):
        """Override to eagerly load relationships for list view"""
        from sqlalchemy.orm import joinedload
        return super().get_list_query(request).options(
            joinedload(ProjectAccountStatus.project),
            joinedload(ProjectAccountStatus.account)
        )
    
    def get_object_query(self, request):
        """Override to eagerly load relationships for detail/edit views"""
        from sqlalchemy.orm import joinedload
        return super().get_object_query(request).options(
            joinedload(ProjectAccountStatus.project),
            joinedload(ProjectAccountStatus.account)
        )


class ProjectLeaderboardHistoryAdmin(AdminWriteModeratorReadView):
    label = "Leaderboard History"
    icon = "fa fa-trophy"
    page_size = 100
    sortable_fields = ["id", "created_at", "start_ts", "end_ts", "project_id"]
    searchable_fields = ["id", "project_id"]
    fields = [
        IntegerField("id", label="ID", read_only=True),
        IntegerField("project_id", label="Project ID", read_only=True),
        IntegerField("start_ts", label="Start TS", read_only=True),
        IntegerField("end_ts", label="End TS", read_only=True),
        IntegerField("created_at", label="Created At", read_only=True),
        HasMany("scores", label="Score Payouts", identity="score-payout"),
    ]
    exclude_fields_from_create = ["id", "created_at", "scores"]
    exclude_fields_from_edit = ["id", "created_at", "scores"]


class ScorePayoutAdmin(AdminWriteModeratorReadView):
    label = "Score Payouts"
    icon = "fa fa-coins"
    page_size = 100
    sortable_fields = ["id", "created_at", "project_id", "social_account_id", "score", "loyalty_points"]
    searchable_fields = ["id", "project_id", "social_account_id"]
    fields = [
        IntegerField("id", label="ID", read_only=True),
        IntegerField("social_account_id", label="Social Account ID", read_only=True),
        IntegerField("project_id", label="Project ID", read_only=True),
        IntegerField("project_leaderboard_history_id", label="Leaderboard History ID", read_only=True),
        IntegerField("created_at", label="Created At", read_only=True),
        FloatField("engagement", label="Engagement", read_only=True),
        FloatField("mindshare", label="Mindshare", read_only=True),
        FloatField("base_score", label="Base Score", read_only=True),
        FloatField("score", label="Score", read_only=True),
        IntegerField("new_posts_count", label="New Posts Count", read_only=True),
        IntegerField("first_post_at", label="First Post At (timestamp)", read_only=True),
        IntegerField("last_post_at", label="Last Post At (timestamp)", read_only=True),
        IntegerField("weekly_streak_start_at", label="Weekly Streak Start At (timestamp)", read_only=True),
        FloatField("loyalty_points", label="Loyalty Points", read_only=True),
        FloatField("min_loyalty", label="Min Loyalty", read_only=True),
    ]
    exclude_fields_from_create = ["id", "created_at"]
    exclude_fields_from_edit = ["id", "created_at"]



# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def setup_admin(app: Any) -> None:  # FastAPI/Starlette app
    """Mount Starlette-Admin to *app* under /admin path."""
    admin = SAAdmin(
        engine,
        title="Sidelined Admin",
        auth_provider=BasicAuthProvider(),
    )

    # Register views with proper labels and icons
    admin.add_view(AdminUserAdmin(AdminUserModel, label="Admin Users", icon="fa fa-user-shield"))
    admin.add_view(UserAdmin(User, label="Users", icon="fa fa-users"))
    admin.add_view(ProjectAdmin(Project, label="Projects", icon="fa fa-project-diagram"))
    admin.add_view(SocialAccountAdmin(SocialAccount, label="Social Accounts", icon="fa fa-share-alt"))
    admin.add_view(ProjectAccountStatusAdmin(ProjectAccountStatus, label="Project Account Status", icon="fa fa-link"))
    admin.add_view(ProjectLeaderboardHistoryAdmin(ProjectLeaderboardHistory, label="Leaderboard History", icon="fa fa-trophy"))
    admin.add_view(ScorePayoutAdmin(ScorePayout, label="Score Payouts", icon="fa fa-coins"))

    # Finally mount to the app
    admin.mount_to(app)

    logger.info("Starlette-Admin initialised successfully") 