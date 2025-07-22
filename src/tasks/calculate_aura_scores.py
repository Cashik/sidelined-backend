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


async def _calculate_aura_scores_async():
    """
    Асинхронная функция для расчета Aura scores.
    Выделена отдельно для правильной работы с event loop.
    """
    with CelerySessionManager() as db:
        processed_count = 0
        projects: List[models.Project] = list(db.execute(select(models.Project)).scalars())
        
        if not projects:
            logger.info("calculate_aura_scores: нет проектов в базе")
            return {"processed_count": 0}
        
        for project in projects:
            if not project.is_leaderboard_project:
                continue
            
            logger.info("calculate_aura_scores: project=%s", project.name)
            try:
                result_count = await utils.calculate_daily_aura_score(project, db)
                logger.info("calculate_aura_scores: project=%s processed %d posts", project.name, result_count)
                processed_count += result_count
            except Exception as e:
                logger.error("calculate_aura_scores: error for project=%s: %s", project.name, str(e), exc_info=True)
                continue
            
        logger.info("calculate_aura_scores: total processed %d posts across all projects", processed_count)
        return {"processed_count": processed_count}


@celery_app.task(
    bind=True,
    name="src.tasks.calculate_aura_scores.calculate_aura_scores",
    max_retries=2,
    default_retry_delay=300,  # 5 минут между повторами
)
def calculate_aura_scores(self):  # type: ignore[override]
    """
    Рассчитывает Aura scores для постов всех проектов.
    
    Эта задача должна запускаться каждый час для расчета Aura scores
    для новых постов за предыдущий период.
    """
    try:
        return asyncio.run(_calculate_aura_scores_async())
    except Exception as exc:
        logger.error("calculate_aura_scores task failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc) 