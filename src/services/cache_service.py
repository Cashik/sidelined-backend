import json
from typing import Any, Optional
from enum import Enum

from src.services.redis_client import redis_client
from src.config.settings import settings

LEADERBOARD_CACHE_VERSION = 1.2
FEED_CACHE_VERSION = 1

class LeaderboardPeriod(str, Enum):
    ONE_DAY = "1d"
    ONE_WEEK = "1w"
    ONE_MONTH = "1m"
    ALL_TIME = "all"
    # можно добавить другие периоды 

class CacheService:
    def __init__(self, prefix: str = "cache"):
        self.prefix = prefix

    def _make_key(self, key: str) -> str:
        return f"{self.prefix}:{key}"

    async def get(self, key: str) -> Optional[Any]:
        redis_key = self._make_key(key)
        data = await redis_client.get(redis_key)
        if data is None:
            return None
        try:
            return json.loads(data)
        except Exception:
            return data

    async def set(self, key: str, value: Any, ttl: int = 300):
        redis_key = self._make_key(key)
        data = json.dumps(value)
        await redis_client.set(redis_key, data, ex=ttl)

    async def delete(self, key: str):
        redis_key = self._make_key(key)
        await redis_client.delete(redis_key)

    async def exists(self, key: str) -> bool:
        redis_key = self._make_key(key)
        return await redis_client.exists(redis_key) > 0

# Пример специализированного сервиса для лидерборда
class LeaderboardCacheService(CacheService):
    def __init__(self, version: int = LEADERBOARD_CACHE_VERSION):
        super().__init__(prefix="leaderboard")
        self.version = version

    def leaderboard_key(self, project_id: int, period: LeaderboardPeriod) -> str:
        return f"{self.version}:{project_id}:{period.value}"

    async def get_leaderboard(self, project_id: int, period: LeaderboardPeriod) -> Optional[Any]:
        return await self.get(self.leaderboard_key(project_id, period))

    async def set_leaderboard(self, project_id: int, period: LeaderboardPeriod, value: Any, ttl: int = int(settings.MASTER_UPDATE_PERIOD_SECONDS*1.5)):
        await self.set(self.leaderboard_key(project_id, period), value, ttl)

    async def delete_leaderboard(self, project_id: int, period: LeaderboardPeriod):
        await self.delete(self.leaderboard_key(project_id, period))

# Новый сервис для кеша фида
class FeedCacheService(CacheService):
    def __init__(self, version: int = FEED_CACHE_VERSION):
        super().__init__(prefix="feed")
        self.version = version

    def feed_key(self, project_id: int) -> str:
        return f"{self.version}:{project_id}"

    async def get_feed(self, project_id: int) -> Optional[Any]:
        return await self.get(self.feed_key(project_id))

    async def set_feed(self, project_id: int, value: Any, ttl: int = int(settings.MASTER_UPDATE_PERIOD_SECONDS*1.5)):
        await self.set(self.feed_key(project_id), value, ttl)

    async def delete_feed(self, project_id: int):
        await self.delete(self.feed_key(project_id))

