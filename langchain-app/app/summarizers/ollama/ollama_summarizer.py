from typing import Any

import ollama
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.caches import BaseCache
from langchain_core.runnables.base import Runnable
from langchain_core.language_models.chat_models import BaseChatModel

from app.summarizers.base.base_summarizer import BaseSummarizer


class OllamaSummarizer(BaseSummarizer):

    def __init__(self, base_url: str, model_name: str, cache: BaseCache = None, **kwargs) -> None:
        self._ensure_model_is_pulled_in_server(model_name=model_name, base_url=base_url)
        self.cache = cache
        self.base_url = base_url
        self.model_name = model_name
        super().__init__(**kwargs)

    @property
    def model(self) -> BaseChatModel:
        return ChatOllama(model=self.model_name, base_url=self.base_url, cache=self.cache)

    def create_runnable(self) -> Runnable:
        # TODO in the future we may wanna set the output parser as a property
        return self.prompt | self.model | StrOutputParser()

    def _ensure_model_is_pulled_in_server(self, model_name: str, base_url: str) -> None:
        client = ollama.Client(host=base_url)
        if (
            not client.list()['models']
            or not any(model_name == m['name'] for m in client.list()['models'])
        ):
            client.pull(model=model_name)

    def get_metadata(self, file_name: str, content) -> dict[str, Any]:
        # TODO incorporate here other info from the returned iterators as metadata
        return {
            'summarizer': self.__class__.__name__,
            'model_name': self.model_name,
            'loader': self.loader.__class__.__name__,
            'file_name': file_name,
            # 'created_at': content.response_metadata['created_at'],
            # 'usage': content.usage_metadata,
            # 'done_reason': content.response_metadata['done_reason'],
        }
