from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from sqlalchemy.types import Enum

from alembic import context

# Импортируем все необходимые модели
from sqlmodel import SQLModel
# Просто импортируем models, он сам зарегистрирует все модели
from src.models import *  # noqa

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

from src.config import settings

# Используем уже готовые метаданные
target_metadata = SQLModel.metadata

def get_url():
    return settings.DATABASE_URL

# Функция для обработки ENUM типов, предотвращающая их пересоздание
def render_item(type_, obj, autogen_context):
    """
    Переопределяем рендеринг для исключения повторного создания ENUM типов
    """
    # Обрабатываем ENUM типы, чтобы не создавать их заново
    if type_ == 'type' and isinstance(obj, Enum):
        return None  # Пропускаем генерацию кода для ENUM типов
    # Используем стандартное поведение для других типов
    return False

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_url()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata,
            render_item=render_item  # Добавляем функцию для обработки ENUM типов
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
