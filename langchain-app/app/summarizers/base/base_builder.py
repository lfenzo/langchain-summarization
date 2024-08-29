from abc import abstractmethod

from langchain_core.caches import BaseCache
from langchain_core.document_loaders import BaseLoader

from app.factories.cache_factory import CacheFactory
from app.factories.loader_factory import LoaderFactory
from app.factories.store_manager_factory import StorageManagerFactory
from app.storage.base_store_manager import BaseStoreManager


class SummarizerBuilder:

    def __init__(self) -> None:
        self.loader = None
        self.cache = CacheFactory().create(cache_type='redis', host='redis', port=6379)
        self.store_manager = StorageManagerFactory().create(
            manager='mongodb',
            database_name='summary_database',
            collection_name='summaries',
            user='root',
            password='examplepassword',  # TODO pass via ENV variables
        )

    @abstractmethod
    def build():
        pass

    def get_init_params(self):
        return {
            'cache': self.cache,
            'loader': self.loader,
            'store_manager': self.store_manager,
        }

    def set_store_manager(self, manager: str, store_manager: BaseStoreManager = None, **kwargs):
        self.store_manager = (
            store_manager if store_manager is not None
            else StorageManagerFactory().create(manager=manager, **kwargs)
        )
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
