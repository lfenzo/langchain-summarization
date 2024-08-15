from langchain_core.caches import BaseCache
from langchain_core.stores import BaseStore, ByteStore
from langchain_core.document_loaders import BaseLoader

from app.summarizers.factories.loader_factory import LoaderFactory
from app.summarizers.factories.cache_factory import CacheFactory


class SummarizerBuilder:

    def __init__(self) -> None:
        self.store = None
        self.loader = None
        self.byte_store = None
        self.cache = CacheFactory().create(cache_type='redis')

    def set_store(self, store: BaseStore):
        self.store = store
        return self

    def set_byte_store(self, byte_store: ByteStore):
        self.byte_store = byte_store
        return self

    def set_loader(self, file_type: str = None, file_path: str = None, loader: BaseLoader = None):
        self.loader = (
            loader if loader is not None
            else LoaderFactory().create(file_type=file_type, file_path=file_path)
        )
        return self

    def set_cache(self, cache_type: str = None, cache: BaseCache = None):
        self.cache = (
            cache if cache is not None
            else CacheFactory().create(cache_type=cache_type)
        )
        return self

    def build(self) -> dict:
        return {
            'store': self.store,
            'loader': self.loader,
            'byte_store': self.byte_store,
            'cache': self.cache,
        }
