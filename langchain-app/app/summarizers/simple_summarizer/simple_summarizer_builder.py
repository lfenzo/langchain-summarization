from langchain_core.caches import BaseCache
from langchain_core.language_models.chat_models import BaseChatModel

from app.factories.cache_factory import CacheFactory
from app.summarizers.base.base_builder import SummarizerBuilder
from app.summarizers.simple_summarizer.simple_summarizer import SimmpleSummarizer
from app.factories.chatmodel_factory import ChatModelFactory


class SimmpleSummarizerBuilder(SummarizerBuilder):
    DEFAULT_CACHE_TYPE = 'redis'
    DEFAULT_CACHE_HOST = 'redis'
    DEFAULT_CACHE_PORT = 6379
    DEFAULT_CHATMODEL_SERVICE = 'ollama'
    DEFAULT_CHATMODEL_KWARGS = {
        'model': 'llama3.1',
        'base_url': 'http://ollama-server:11434',
    }

    def __init__(self) -> None:
        super().__init__()
        self.chatmodel = None
        self.has_system_msg_support = False
        self.cache = self._create_default_cache()
        self._chatmodel_service = self.DEFAULT_CHATMODEL_SERVICE
        self._chatmodel_kwargs = self.DEFAULT_CHATMODEL_KWARGS.copy()

    def build(self) -> SimmpleSummarizer:
        self._create_chatmodel()
        return SimmpleSummarizer(**self.get_init_params())

    def get_init_params(self) -> dict:
        params = {
            "chatmodel": self.chatmodel,
            "has_system_msg_support": self.has_system_msg_support,
        }
        params.update(super().get_init_params())
        return params

    def set_cache(self, cache: str | BaseCache, **kwargs):
        self.cache = (
            cache if isinstance(cache, BaseCache)
            else CacheFactory().create(cache=cache, **kwargs)
        )
        return self

    def set_chatmodel(self, chatmodel: str | BaseChatModel, **kwargs):
        if isinstance(chatmodel, BaseChatModel):
            self.chatmodel = chatmodel
        else:
            self._chatmodel_service = chatmodel
            # 'update' as we may not provide all necessary kwargs when calling set_chatmodel
            self._chatmodel_kwargs.update(kwargs)
        return self

    def set_system_msg_support(self, has_system_msg_support: bool):
        self.has_system_msg_support = has_system_msg_support
        return self

    def _create_default_cache(self) -> BaseCache:
        return CacheFactory().create(
            cache=self.DEFAULT_CACHE_TYPE,
            host=self.DEFAULT_CACHE_HOST,
            port=self.DEFAULT_CACHE_PORT
        )

    def _create_chatmodel(self) -> None:
        if not self.chatmodel:
            self.chatmodel = ChatModelFactory().create(
                chatmodel=self._chatmodel_service,
                cache=self.cache,
                **self._chatmodel_kwargs
            )
