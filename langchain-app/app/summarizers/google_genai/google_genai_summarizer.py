from typing import Any

from langchain.prompts import ChatPromptTemplate
from langchain_core.caches import BaseCache
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables.base import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel

from app.summarizers.base.base_summarizer import BaseSummarizer


class GoogleGenAISummarizer(BaseSummarizer):

    def __init__(self, model_name: str, location: str, cache: BaseCache = None, **kwargs) -> None:
        self.cache = cache
        self.location = location
        self.model_name = model_name
        super().__init__(**kwargs)

    @property
    def prompt(self):
        # Google GenerativeAI does not support system messagem the way Ollama does
        # the base prompt is rewritten here replacing "system" messages with "human" ones.
        return ChatPromptTemplate.from_messages([
            ('human', "You are an expert multi-language AI summary writer."),
            ('human', "Produce a summary of the provided text."),
            ('human', "Do not provide an introduction, just the summary."),
            ('human', "The summary must contain ~25% of the length of the original"),
            ('human', "Summary language must be the same as the original"),
            ('human', "Tailor the summary to what you assume to be the document audience"),
            ('human', "Don't ask for follouw-up questions."),
            ('human', "{text}"),
        ])

    def create_runnable(self) -> Runnable:
        return self.prompt | self.model | StrOutputParser()

    @property
    def model(self) -> BaseChatModel:
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            cache=self.cache,
            location=self.location
        )

    def get_metadata(self, file_name: str, content) -> dict[str, Any]:
        return {
            'summarizer': self.__class__.__name__,
            'model_name': self.model_name,
            'loader': self.loader.__class__.__name__,
            'file_name': file_name,
        }
