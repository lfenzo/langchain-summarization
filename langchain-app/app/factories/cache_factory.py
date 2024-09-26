from redis import Redis

from langchain_core.caches import BaseCache
from langchain_community.cache import RedisCache


class CacheFactory:
    """
    Factory class for creating cache instances.

    Attributes
    ----------
    available_caches : dict
        A dictionary mapping cache types (str) to their respective factory methods.
    """

    def __init__(self):
        self.available_caches = {
            'redis': self._get_redis_cache,
        }

    def create(self, cache: str, **kwargs) -> BaseCache:
        """
        Create a cache instance based on the specified cache type.

        Parameters
        ----------
        cache : str
            The cache type to create (e.g., 'redis').
        **kwargs : dict
            Additional keyword arguments passed to the cache factory method.

        Returns
        -------
        BaseCache
            The cache instance created.

        Raises
        ------
        ValueError
            If the specified cache type is not valid.

        Examples
        --------
        >>> cache_factory = CacheFactory()
        >>> redis_cache = cache_factory.create('redis', host='localhost', port=6379)
        """
        if cache not in self.available_caches:
            raise ValueError(
                f"Invalid cache type '{cache}'. "
                f"Valid cache types are: {self.get_valid_cache_types()}"
            )
        return self.available_caches[cache](**kwargs)

    def _get_redis_cache(
        self,
        host: str,
        port: int,
        decode_responses: bool = True,
        **kwargs
    ) -> RedisCache:
        """
        Creates a Redis cache instance.

        Parameters
        ----------
        host : str
            The Redis server hostname.
        port : int
            The Redis server port.
        decode_responses : bool, optional
            Whether to decode responses from Redis (default is True).
        **kwargs : dict
            Additional keyword arguments for configuring the Redis cache.

        Returns
        -------
        RedisCache
            A RedisCache instance.
        """
        return RedisCache(
            redis_=Redis(host=host, port=port, decode_responses=decode_responses),
            **kwargs,
        )

    def get_valid_cache_types(self) -> list[str]:
        """
        Get a list of valid cache types that can be created.

        Returns
        -------
        list[str]
            A list of valid cache type keys.
        """
        return list(self.available_caches.keys())
