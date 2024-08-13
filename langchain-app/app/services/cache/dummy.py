from app.services.cache.base_cache import BaseCache


class DummyCache(BaseCache):

    def __init__(self):
        print("Cache initialized")

    def set(self, key: str, value):
        return False

    def get(self, key: str):
        return False
