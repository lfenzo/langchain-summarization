from typing import Any

from langchain_core.caches import BaseCache
from langchain_google_vertexai import ChatVertexAI
from langchain_core.language_models.chat_models import BaseChatModel

from app.summarizers.base.base_summarizer import BaseSummarizer


class GoogleVertexAISummarizer(BaseSummarizer):

    def __init__(self, model_name: str, location: str, cache: BaseCache = None, **kwargs) -> None:
        self.cache = cache
        self.location = location
        self.model_name = model_name
        super().__init__(**kwargs)

    @property
    def model(self) -> BaseChatModel:
        return ChatVertexAI(model=self.model_name, cache=self.cache, location=self.location)

    def get_metadata(self, file_name: str, content, last_chunk: dict) -> dict[str, Any]:
        return {
            'summarizer': self.__class__.__name__,
            'loader': self.loader.__class__.__name__,
            'file_name': file_name,
            **last_chunk.response_metadata,
            **last_chunk.usage_metadata,
        }
