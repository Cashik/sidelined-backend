from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
import logging
import time
from sqlalchemy.exc import OperationalError
from typing import Generator

from src.config.settings import settings

logger = logging.getLogger(__name__)

def create_db_engine():
    """Создание движка базы данных с повторными попытками подключения"""
    logger.info(f"Attempting to connect to database at {settings.DB_HOST}:{settings.DB_PORT}")
    logger.debug(f"Database URL: {settings.DATABASE_URL}")
    
    # Базовые параметры движка
    engine_params = {
        "echo": False,  # Выключаем SQL логирование в режиме отладки
    }
    
    # Если это не тестовое окружение (PostgreSQL)
    if not settings.TESTING:
        engine_params.update({
            "pool_size": 20,
            "max_overflow": 20,
            "pool_timeout": 60,
            "pool_recycle": 3600,
            "pool_pre_ping": True
        })
    # Для SQLite в тестах
    else:
        engine_params.update({
            "connect_args": {"check_same_thread": False}
        })
    
    retries = 5
    while retries > 0:
        try:
            logger.debug(f"Creating database engine (attempt {6-retries}/5)")
            engine = create_engine(settings.DATABASE_URL, **engine_params)
            
            # Проверяем подключение
            logger.debug("Testing database connection...")
            with engine.connect() as conn:
                if not settings.TESTING:
                    result = conn.execute(text("SELECT version()")).scalar()
                    logger.info(f"Successfully connected to PostgreSQL. Version: {result}")
                else:
                    result = conn.execute(text("SELECT sqlite_version()")).scalar()
                    logger.info(f"Successfully connected to SQLite. Version: {result}")
            return engine
        except OperationalError as e:
            retries -= 1
            if retries == 0:
                logger.error(f"Failed to connect to database after 5 attempts: {str(e)}")
                raise
            if not settings.TESTING:  # Повторные попытки только для PostgreSQL
                logger.warning(f"Failed to connect to database: {str(e)}. Retrying in 5 seconds... ({retries} attempts left)")
                time.sleep(5)
            else:
                raise
        except Exception as e:
            logger.error(f"Unexpected error while connecting to database: {str(e)}")
            raise

# Создаем движок при импорте модуля
try:
    engine = create_db_engine()
except Exception as e:
    logger.error(f"Failed to create database engine: {str(e)}")
    raise

# Создаем фабрику сессий
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_session() -> Generator[Session, None, None]:
    """Получение сессии базы данных с логированием состояния пула"""
    try:
        session = SessionLocal()
        if not settings.TESTING:  # Логируем состояние пула только для PostgreSQL
            logger.debug(
                f"DB Pool Status - Checked in: {engine.pool.checkedin()}, "
                f"Checked out: {engine.pool.checkedout()}, "
                f"Size: {engine.pool.size()}"
            )
        try:
            yield session
        finally:
            session.close()
    except Exception as e:
        logger.error(f"Database session error: {str(e)}")
        raise
    
