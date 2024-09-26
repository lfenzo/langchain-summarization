from abc import ABC, abstractmethod

from langchain_core.caches import BaseCache
from langchain_core.document_loaders import BaseLoader
from langchain_core.language_models.chat_models import BaseChatModel

from app.factories import (
    CacheFactory,
    ChatModelFactory,
    ExecutionStrategyFactory,
    LoaderFactory,
    StoreManagerFactory,
)
from app.storage import BaseStoreManager
from app.strategies.execution import BaseExecutionStrategy


class BaseBuilder(ABC):
    """
    Abstract base class for building various components such as loaders, caches, store managers,
    and execution strategies for the Summarizer object.
    """

    DEFAULT_CACHE_SERVICE = 'redis'
    DEFAULT_CACHE_HOST = 'redis'
    DEFAULT_CACHE_PORT = 6379
    DEFAULT_STORE_MANAGER_SERVICE = 'mongodb'

    def __init__(self) -> None:
        """
        Initializes the BaseBuilder with default cache, store manager and execution strategy.

        Sets up default instances of the cache, store manager, and execution strategy.
        """
        self.loader = None
        self.cache = self._create_default_cache()
        self.store_manager = self._create_default_store_manager()
        self.execution_strategy = self._create_default_execution_strategy()

    @abstractmethod
    def build():
        """Abstract method for building the necessary components."""
        pass

    def get_init_params(self):
        """
        Retrieves the initialized parameters required to create the summarizer or other components.

        Returns
        -------
        dict
            A dictionary containing the loader, store manager, and execution strategy.
        """
        return {
            'loader': self.loader,
            'store_manager': self.store_manager,
            'execution_strategy': self.execution_strategy,
        }

    def set_store_manager(self, store_manager: str | BaseStoreManager, **kwargs):
        """
        Sets the store manager, either by creating a new instance or using an existing one.

        Parameters
        ----------
        store_manager : str or BaseStoreManager
            The name of the store manager service or an instance of BaseStoreManager.
        **kwargs : dict
            Additional keyword arguments for creating a new store manager instance.

        Returns
        -------
        BaseBuilder
            Returns the current instance of BaseBuilder for method chaining.
        """
        self.store_manager = (
            store_manager if isinstance(store_manager, BaseStoreManager)
            else StoreManagerFactory().create(store_manager=store_manager, **kwargs)
        )
        return self

    def set_cache(self, cache: str | BaseCache, **kwargs):
        """
        Sets the cache, either by creating a new instance or using an existing one.

        Parameters
        ----------
        cache : str or BaseCache
            The name of the cache service or an instance of BaseCache.
        **kwargs : dict
            Additional keyword arguments for creating a new cache instance.

        Returns
        -------
        BaseBuilder
            Returns the current instance of BaseBuilder for method chaining.
        """
        self.cache = (
            cache if isinstance(cache, BaseCache)
            else CacheFactory().create(cache=cache, **kwargs)
        )
        return self

    def set_loader(self, file_type: str = None, file_path: str = None, loader: BaseLoader = None):
        """
        Sets the loader, either by creating a new instance or using an existing one.

        Parameters
        ----------
        file_type : str, optional
            The type of file to load (default is None).
        file_path : str, optional
            The path to the file to load (default is None).
        loader : BaseLoader, optional
            An instance of BaseLoader (default is None).

        Returns
        -------
        BaseBuilder
            Returns the current instance of BaseBuilder for method chaining.
        """
        self.loader = (
            loader if loader is not None
            else LoaderFactory().create(file_type=file_type, file_path=file_path)
        )
        return self

    def set_execution_strategy(self, execution_strategy: str | BaseExecutionStrategy):
        """
        Sets the execution strategy, either by creating a new instance or using an existing one.

        Parameters
        ----------
        execution_strategy : str or BaseExecutionStrategy
            The name of the execution strategy service or an instance of BaseExecutionStrategy.

        Returns
        -------
        BaseBuilder
            Returns the current instance of BaseBuilder for method chaining.
        """
        self.execution_strategy = (
            execution_strategy if isinstance(execution_strategy, BaseExecutionStrategy)
            else ExecutionStrategyFactory().create(strategy=execution_strategy)
        )
        return self

    def _create_chatmodel(self, service: str, chatmodel: BaseChatModel = None, **kwargs):
        """
        Creates or retrieves a chat model, either by creating a new instance or using an existing one.

        Parameters
        ----------
        service : str
            The name of the chat model service to create.
        chatmodel : BaseChatModel, optional
            An existing instance of BaseChatModel (default is None).
        **kwargs : dict
            Additional keyword arguments for creating a new chat model instance.

        Returns
        -------
        BaseChatModel
            The chat model instance created or retrieved.
        """
        return (
            chatmodel if isinstance(chatmodel, BaseChatModel)
            else ChatModelFactory().create(chatmodel=service, cache=self.cache, **kwargs)
        )

    def _create_default_store_manager(self) -> BaseStoreManager:
        return StoreManagerFactory().create(store_manager=self.DEFAULT_STORE_MANAGER_SERVICE)

    def _create_default_cache(self) -> BaseCache:
        return CacheFactory().create(
            cache=self.DEFAULT_CACHE_SERVICE,
            host=self.DEFAULT_CACHE_HOST,
            port=self.DEFAULT_CACHE_PORT
        )

    def _create_default_execution_strategy(self) -> BaseExecutionStrategy:
        return ExecutionStrategyFactory().create(strategy='stream')
