from redis import Redis

from langchain_core.caches import BaseCache
from langchain_community.cache import RedisCache


class CacheFactory:

    def __init__(self):
        self.available_caches = {
            'redis': self._get_redis_cache,
        }

    def create(self, cache_type: str) -> BaseCache:
        return self.available_caches[cache_type]()

    def _get_redis_cache(self):
        return RedisCache(redis_=Redis(host='redis', port=6379, decode_responses=True), ttl=60)
