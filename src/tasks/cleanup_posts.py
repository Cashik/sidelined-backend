from __future__ import annotations

import asyncio
import logging
from sqlalchemy.orm import Session

from src.core.celery_app import celery_app
from src.database import SessionLocal
from src import crud, utils_base
from src.config.settings import settings

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="src.tasks.cleanup_posts.cleanup_old_posts",
    max_retries=3,
    default_retry_delay=300,  # 5 минут между повторами
)
def cleanup_old_posts(self):  # type: ignore[override]
    """
    Удаляет все посты из базы данных, которые старше чем POST_INACTIVE_PERIOD_SECONDS.
    
    Эта задача должна запускаться периодически для очистки устаревших данных
    и предотвращения неконтролируемого роста базы данных.
    """
    db: Session = SessionLocal()
    try:
        # Вычисляем timestamp, старше которого посты считаются неактуальными
        cutoff_timestamp = utils_base.now_timestamp() - settings.POST_INACTIVE_TIME_SECONDS
        
        logger.info(
            "cleanup_old_posts: starting cleanup for posts older than %d (cutoff: %d seconds ago)",
            cutoff_timestamp,
            settings.POST_INACTIVE_TIME_SECONDS
        )
        
        # Удаляем старые посты
        deleted_count = asyncio.run(crud.delete_old_posts(db, cutoff_timestamp))
        
        logger.info(
            "cleanup_old_posts: successfully deleted %d old posts (older than %d)",
            deleted_count,
            cutoff_timestamp
        )
        
        return {
            "deleted_count": deleted_count,
            "cutoff_timestamp": cutoff_timestamp,
            "cutoff_period_seconds": settings.POST_INACTIVE_TIME_SECONDS
        }
        
    except Exception as exc:
        logger.error("cleanup_old_posts task failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc)
    finally:
        db.close() 