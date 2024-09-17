from abc import ABC, abstractmethod

from langchain_core.caches import BaseCache
from langchain_core.document_loaders import BaseLoader
from langchain_core.language_models.chat_models import BaseChatModel

from app.factories.cache_factory import CacheFactory
from app.factories.chatmodel_factory import ChatModelFactory
from app.factories.loader_factory import LoaderFactory
from app.factories.store_manager_factory import StoreManagerFactory
from app.storage.base_store_manager import BaseStoreManager


class BaseBuilder(ABC):
    DEFAULT_CACHE_SERVICE = 'redis'
    DEFAULT_CACHE_HOST = 'redis'
    DEFAULT_CACHE_PORT = 6379
    DEFAULT_STORE_MANAGER_SERVICE = 'mongodb'

    def __init__(self) -> None:
        self.loader = None
        self.cache = self._create_default_cache()
        self.store_manager = self._create_default_store_manager()

    @abstractmethod
    def build():
        pass

    def get_init_params(self):
        return {
            'loader': self.loader,
            'store_manager': self.store_manager,
        }

    def set_store_manager(self, store_manager: str | BaseStoreManager, **kwargs):
        self.store_manager = (
            store_manager if isinstance(store_manager, BaseStoreManager)
            else StoreManagerFactory().create(store_manager=store_manager, **kwargs)
        )
        return self

    def set_cache(self, cache: str | BaseCache, **kwargs):
        self.cache = (
            cache if isinstance(cache, BaseCache)
            else CacheFactory().create(cache=cache, **kwargs)
        )
        return self

    def set_loader(self, file_type: str = None, file_path: str = None, loader: BaseLoader = None):
        self.loader = (
            loader if loader is not None
            else LoaderFactory().create(file_type=file_type, file_path=file_path)
        )
        return self

    def _create_chatmodel(self, service: str, chatmodel: BaseChatModel = None, **kwargs):
        return (
            chatmodel if isinstance(chatmodel, BaseChatModel)
            else ChatModelFactory().create(chatmodel=service, cache=self.cache, **kwargs)
        )

    def _create_default_store_manager(self):
        return StoreManagerFactory().create(store_manager=self.DEFAULT_STORE_MANAGER_SERVICE)

    def _create_default_cache(self) -> BaseCache:
        return CacheFactory().create(
            cache=self.DEFAULT_CACHE_SERVICE,
            host=self.DEFAULT_CACHE_HOST,
            port=self.DEFAULT_CACHE_PORT
        )
