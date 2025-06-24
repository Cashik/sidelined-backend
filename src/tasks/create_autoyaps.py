from __future__ import annotations

import asyncio
import logging
from typing import List
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.core.celery_app import celery_app
from src.database import CelerySessionManager
from src import utils, models
from src.config.settings import settings

logger = logging.getLogger(__name__)


async def _create_autoyaps_async():
    """
    Асинхронная функция для создания авто-постов.
    Выделена отдельно для правильной работы с event loop.
    """
    with CelerySessionManager() as db:
        created_count = 0
        projects: List[models.Project] = list(db.execute(select(models.Project)).scalars())
        if not projects:
            logger.info("create_autoyaps: нет проектов в базе")
            return {"created_count": 0}
        
        for project in projects:
            logger.info("create_autoyaps: project=%s", project.name)
            try:
                result = await utils.create_project_autoyaps(project, db)
                logger.info("create_autoyaps: project=%s done => %s", project.name, len(result))
                created_count += len(result)
            except Exception as e:
                logger.error("create_autoyaps: error for project=%s: %s", project.name, str(e))
                continue
            
        return {"created_count": created_count}


@celery_app.task(
    bind=True,
    name="src.tasks.create_autoyaps.create_autoyaps",
    max_retries=0,
    default_retry_delay=300,  # 5 минут между повторами
)
def create_autoyaps(self):  # type: ignore[override]
    """
    Создает авто-посты для всех проектов.
    
    Эта задача должна запускаться периодически для создания авто-постов.
    """
    try:
        return asyncio.run(_create_autoyaps_async())
    except Exception as exc:
        logger.error("create_autoyaps task failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc) 