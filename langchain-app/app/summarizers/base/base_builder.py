from abc import abstractmethod

from langchain_core.document_loaders import BaseLoader

from app.factories.loader_factory import LoaderFactory
from app.factories.store_manager_factory import StoreManagerFactory
from app.storage.base_store_manager import BaseStoreManager


class SummarizerBuilder:

    def __init__(self) -> None:
        self.loader = None
        self.store_manager = StoreManagerFactory().create(store_manager='mongodb')

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

    def set_loader(self, file_type: str = None, file_path: str = None, loader: BaseLoader = None):
        self.loader = (
            loader if loader is not None
            else LoaderFactory().create(file_type=file_type, file_path=file_path)
        )
        return self
