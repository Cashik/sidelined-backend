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


async def _master_update_async():
    """
    Асинхронная функция для синхронизации постов.
    Выделена отдельно для правильной работы с event loop.
    """
    with CelerySessionManager() as db:
        await utils.master_update(db)


@celery_app.task(
    bind=True,
    name="src.tasks.master_update.master_update",
    max_retries=3,
    default_retry_delay=60,
)
def master_update(self):  # type: ignore[override]
    try:
        asyncio.run(_master_update_async())
    except Exception as exc:
        logger.error("master_update task failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc) 