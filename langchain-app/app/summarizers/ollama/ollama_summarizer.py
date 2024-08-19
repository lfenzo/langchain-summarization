from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.caches import BaseCache
from langchain_core.messages import AIMessage

from app.summarizers.base.base_summarizer import BaseSummarizer


class OllamaSummarizer(BaseSummarizer):

    def __init__(
        self,
        base_url: str,
        model: str,
        cache: BaseCache = None,
        **kwargs,
    ):
        self.model = model
        self.base_url = base_url
        runnable = self.prompt | ChatOllama(model=self.model, base_url=self.base_url, cache=cache)
        super().__init__(runnable=runnable, **kwargs)

    def render_summary(self, content):
        text = ""
        for page in content:
            text += page.page_content + "\n"
        return self.runnable.invoke(text)

    def get_metadata(self, file_name: str, content: AIMessage) -> dict[str, Any]:
        return {
            'summarizer': self.__class__.__name__,
            'model': content.response_metadata['model'],
            'loader': self.loader.__class__.__name__,
            'file_name': file_name,
            'created_at': content.response_metadata['created_at'],
            'usage': content.usage_metadata,
            'done_reason': content.response_metadata['done_reason'],
        }