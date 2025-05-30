from __future__ import annotations

import asyncio
import logging
from typing import List

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.core.celery_app import celery_app
from src.database import CelerySessionManager
from src import models, utils, utils_base
from src.config.settings import settings
logger = logging.getLogger(__name__)


async def _sync_posts_async():
    """
    Асинхронная функция для синхронизации постов.
    Выделена отдельно для правильной работы с event loop.
    """
    with CelerySessionManager() as db:
        projects: List[models.Project] = list(db.execute(select(models.Project)).scalars())
        if not projects:
            logger.info("sync_posts: нет проектов в базе")
            return

        logger.info("sync_posts: start for %d projects", len(projects))
        from_ts: int = utils_base.now_timestamp() - settings.POST_SYNC_PERIOD_SECONDS  # 3 дня назад

        for project in projects:
            try:
                logger.info("sync_posts: project=%s", project.name)
                result = await utils.update_project_data(project, from_ts, db)
                logger.info("sync_posts: project=%s done => %s", project.name, result)
            except Exception as e:
                logger.error("sync_posts: error on project %s: %s", project.name, e, exc_info=True)
                continue  # переходим к следующему проекту


@celery_app.task(
    bind=True,
    name="src.tasks.sync_posts.sync_posts",
    max_retries=3,
    default_retry_delay=120,
)
def sync_posts(self):  # type: ignore[override]
    """Сканирует ВСЕ проекты и синхронно обновляет посты для каждого.

    Такой подход проще, исключает гонки и превышение rate-limit: мы обрабатываем
    проекты последовательно в рамках одной задачи (один HTTP-коннект к Twitter за раз).
    """
    try:
        asyncio.run(_sync_posts_async())
    except Exception as exc:
        logger.error("sync_posts task failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc) 