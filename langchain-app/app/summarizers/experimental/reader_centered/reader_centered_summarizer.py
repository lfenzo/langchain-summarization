from langchain_core.runnables.base import Runnable
from langchain_core.output_parsers import StrOutputParser

from app.summarizers.ollama.ollama_summarizer import OllamaSummarizer


class ReaderCenteredSummarizer(OllamaSummarizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_runnable(self) -> Runnable:
        return self.prompt | self.model | StrOutputParser()
