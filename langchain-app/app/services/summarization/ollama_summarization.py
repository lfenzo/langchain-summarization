from langchain_ollama import ChatOllama
from langchain_core.caches import BaseCache

from app.services.summarization.base_summarization import BaseSummarization


class OllamaSummarization(BaseSummarization):

    def __init__(
        self,
        base_url: str,
        model: str = "gemma2:27b",
        cache: BaseCache = None,
        **kwargs,
    ):
        self.model = model
        self.base_url = base_url
        runnable = (
            self.prompt
            | ChatOllama(model=self.model, base_url=self.base_url, cache=cache)
        )
        super().__init__(runnable=runnable, **kwargs)

    def render_summary(self, content):
        text = ""
        for page in content:
            text += page.page_content + "\n"
        return self.runnable.invoke(text)
