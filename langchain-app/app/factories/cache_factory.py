from redis import Redis

from langchain_core.caches import BaseCache
from langchain_community.cache import RedisCache


class CacheFactory:

    def __init__(self):
        self.available_caches = {
            'redis': self._get_redis_cache,
        }

    def create(self, cache: str, **kwargs) -> BaseCache:
        return self.available_caches[cache](**kwargs)

    def _get_redis_cache(
        self,
        host: str,
        port: int,
        ttl: int = 60,
        decode_responses: bool = True,
        **kwargs,
    ):
        return RedisCache(
            redis_=Redis(host=host, port=port, decode_responses=decode_responses),
            ttl=ttl,
        )
