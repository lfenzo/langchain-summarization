from langchain_ollama import ChatOllama
from langchain_core.caches import BaseCache

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
