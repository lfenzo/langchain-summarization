from langchain_core.caches import BaseCache
from langchain_core.stores import BaseStore, ByteStore
from langchain_core.document_loaders import BaseLoader

from app.summarizers.loader_factory import LoaderFactory


class SummarizerBuilder:

    def __init__(self) -> None:
        self.store = None
        self.loader = None
        self.byte_store = None
        self.cache = None

    def set_store(self, store: BaseStore):
        self.store = store
        return self

    def set_loader(self, file_type: str, file_path: str, loader: BaseLoader = None):
        self.loader = (
            loader if loader is not None
            else LoaderFactory().create(file_type=file_type, file_path=file_path)
        )
        return self

    def set_byte_store(self, byte_store: ByteStore):
        self.byte_store = byte_store
        return self

    def set_cache(self, cache: BaseCache):
        self.cache = cache
        return self

    def build(self) -> dict:
        return {
            'store': self.store,
            'loader': self.loader,
            'byte_store': self.byte_store,
            'cache': self.cache,
        }
