from typing import Any

import ollama
from langchain_ollama import ChatOllama
from langchain_core.caches import BaseCache
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

    def _ensure_model_is_pulled_in_server(self, model_name: str, base_url: str) -> None:
        client = ollama.Client(host=base_url)
        should_pull_selected_model = (
            not client.list()['models']
            or not any(model_name == m['name'] for m in client.list()['models'])
        )
        if should_pull_selected_model:
            client.pull(model=model_name)

    def get_metadata(self, file_name: str, content, last_chunk: dict) -> dict[str, Any]:
        return {
            'summarizer': self.__class__.__name__,
            'loader': self.loader.__class__.__name__,
            'file_name': file_name,
            **last_chunk.response_metadata,
            **last_chunk.usage_metadata,
        }
