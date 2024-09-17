from langchain_core.language_models.chat_models import BaseChatModel

from app.summarizers.base.base_builder import BaseBuilder
from app.summarizers.experimental.dynamic_prompts.dynamic_prompts_summarizer import (
    DynamicPromptSummarizer
)


class DynamicPromptSummarizerBuilder(BaseBuilder):
    DEFAULT_CHATMODEL_SERVICE = 'ollama'
    DEFAULT_CHATMODEL_KWARGS = {
        'model': 'llama3.1',
        'base_url': 'http://ollama-server:11434',
    }
    DEFAULT_EXTRACTION_CHATMODEL_SERVICE = 'google-genai'
    DEFAULT_EXTRACTION_CHATMODEL_KWARGS = {
        'model': 'gemini-1.5-flash',
        'temperature': 0,
    }

    def __init__(self) -> None:
        super().__init__()
        self.chatmodel = self._create_default_chatmodel()
        self.extraction_chatmodel = self._create_default_extraction_chatmodel()

    def build(self) -> DynamicPromptSummarizer:
        return DynamicPromptSummarizer(**self.get_init_params())

    def get_init_params(self) -> dict:
        params = {
            "chatmodel": self.chatmodel,
            "extraction_chatmodel": self.extraction_chatmodel,
        }
        params.update(super().get_init_params())
        return params

    def set_chatmodel(self, service: str, chatmodel: BaseChatModel = None, **kwargs):
        combined_kwargs = {**self.DEFAULT_CHATMODEL_KWARGS, **kwargs}
        self.chatmodel = self._create_chatmodel(
            service=service, chatmodel=chatmodel, **combined_kwargs
        )
        return self

    def set_extraction_chatmodel(self, service: str, chatmodel: BaseChatModel = None, **kwargs):
        combined_kwargs = {**self.DEFAULT_EXTRACTION_CHATMODEL_KWARGS, **kwargs}
        self.extraction_chatmodel = self._create_chatmodel(
            service=service, chatmodel=chatmodel, **combined_kwargs
        )
        return self

    def _create_default_chatmodel(self) -> BaseChatModel:
        return self._create_chatmodel(
            service=self.DEFAULT_CHATMODEL_SERVICE, **self.DEFAULT_CHATMODEL_KWARGS
        )

    def _create_default_extraction_chatmodel(self) -> BaseChatModel:
        return self._create_chatmodel(
            service=self.DEFAULT_EXTRACTION_CHATMODEL_SERVICE,
            **self.DEFAULT_EXTRACTION_CHATMODEL_KWARGS
        )
