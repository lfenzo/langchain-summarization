from langchain_ollama import ChatOllama

from app.services.summarization.base_summarization import BaseSummarization


class OllamaSummarization(BaseSummarization):

    def __init__(self, base_url: str, model: str = "gemma2:27b", **kwargs):
        self.model = model
        self.base_url = base_url
        runnable = (
            self.prompt | ChatOllama(model=self.model, base_url=self.base_url)
        )
        super().__init__(runnable=runnable, **kwargs)

    def render_summary(self, content):
        print(type(content[0]))
        return self.runnable.invoke({"text": content})
