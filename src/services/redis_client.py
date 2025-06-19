import redis.asyncio as redis
from src.config.settings import settings
from urllib.parse import urlparse
import ssl

parsed_url = urlparse(settings.REDIS_URL)
is_redis_ssl = parsed_url.scheme == 'rediss'

redis_kwargs = {"decode_responses": True}

if is_redis_ssl:
    ssl_cert_reqs_map = {
        'CERT_NONE': ssl.CERT_NONE,
        'CERT_OPTIONAL': ssl.CERT_OPTIONAL,
        'CERT_REQUIRED': ssl.CERT_REQUIRED,
    }
    cert_reqs = ssl_cert_reqs_map.get(settings.REDIS_SSL_CERT_REQS, ssl.CERT_NONE)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = settings.REDIS_SSL_CHECK_HOSTNAME
    ssl_context.verify_mode = cert_reqs
    redis_kwargs["ssl"] = ssl_context

redis_client = redis.from_url(settings.REDIS_URL, **redis_kwargs) 