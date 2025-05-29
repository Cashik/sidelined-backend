from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
import logging
import time
from sqlalchemy.exc import OperationalError
from typing import Generator
import threading

from src.config.settings import settings

logger = logging.getLogger(__name__)

# Глобальная блокировка для thread-safe операций
_engine_lock = threading.Lock()
_celery_engine = None

def create_db_engine(for_celery: bool = False):
    """Создание движка базы данных с повторными попытками подключения"""
    logger.info(f"Attempting to connect to database at {settings.DB_HOST}:{settings.DB_PORT}")
    logger.debug(f"Database URL: {settings.DATABASE_URL}")
    
    # Базовые параметры движка
    engine_params = {
        "echo": False,  # Выключаем SQL логирование в режиме отладки
    }
    
    # Если это для Celery - используем NullPool для изоляции соединений
    if for_celery:
        engine_params.update({
            "poolclass": NullPool,  # Не используем пул соединений для Celery
            "connect_args": {"connect_timeout": 30} if not settings.TESTING else {"check_same_thread": False}
        })
        logger.debug("Creating Celery database engine with NullPool")
    # Если это не тестовое окружение (PostgreSQL)
    elif not settings.TESTING:
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

def get_celery_engine():
    """Получение или создание движка для Celery задач"""
    global _celery_engine
    with _engine_lock:
        if _celery_engine is None:
            _celery_engine = create_db_engine(for_celery=True)
        return _celery_engine

def reset_celery_engine():
    """Сброс движка Celery (используется при форке процессов)"""
    global _celery_engine
    with _engine_lock:
        if _celery_engine is not None:
            try:
                _celery_engine.dispose()
            except Exception as e:
                logger.warning(f"Error disposing Celery engine: {e}")
            _celery_engine = None
    logger.info("Celery database engine reset")

# Создаем основной движок при импорте модуля
try:
    engine = create_db_engine()
except Exception as e:
    logger.error(f"Failed to create database engine: {str(e)}")
    raise

# Создаем фабрику сессий для основного движка
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

class CelerySessionManager:
    """Контекстный менеджер для управления сессиями Celery"""
    
    def __init__(self):
        self.session = None
        
    def __enter__(self) -> Session:
        try:
            # Получаем изолированный движок для Celery
            celery_engine = get_celery_engine()
            # Создаем фабрику сессий для Celery движка
            CelerySessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=celery_engine)
            self.session = CelerySessionLocal()
            logger.debug("Created new isolated Celery session")
            return self.session
        except Exception as e:
            logger.error(f"Failed to create Celery session: {str(e)}")
            raise
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            try:
                if exc_type is not None:
                    # При ошибке откатываем транзакцию
                    self.session.rollback()
                    logger.debug("Rolled back Celery session due to exception")
                else:
                    # При успехе коммитим
                    self.session.commit()
                    logger.debug("Committed Celery session")
            except Exception as e:
                logger.error(f"Error handling Celery session cleanup: {e}")
                try:
                    self.session.rollback()
                except:
                    pass
            finally:
                try:
                    self.session.close()
                    logger.debug("Closed Celery session")
                except Exception as e:
                    logger.error(f"Error closing Celery session: {e}")

def get_celery_session() -> Session:
    """
    DEPRECATED: Используйте CelerySessionManager для безопасного управления сессиями
    Создает новую сессию специально для Celery задач.
    """
    logger.warning("get_celery_session() is deprecated, use CelerySessionManager instead")
    try:
        celery_engine = get_celery_engine()
        CelerySessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=celery_engine)
        session = CelerySessionLocal()
        logger.debug("Created new Celery session (deprecated method)")
        return session
    except Exception as e:
        logger.error(f"Failed to create Celery session: {str(e)}")
        raise

    