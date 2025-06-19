import redis.asyncio as redis
from src.config.settings import settings
from urllib.parse import urlparse
import ssl

redis_kwargs = {"decode_responses": True}

url_without_params = settings.REDIS_URL.split("?")[0]

redis_client = redis.from_url(url_without_params, **redis_kwargs) 