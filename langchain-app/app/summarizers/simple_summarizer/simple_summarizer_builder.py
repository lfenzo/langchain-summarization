from langchain_core.language_models.chat_models import BaseChatModel

from app.summarizers.base.base_builder import BaseBuilder
from app.summarizers.simple_summarizer.simple_summarizer import SimmpleSummarizer


class SimmpleSummarizerBuilder(BaseBuilder):
    DEFAULT_CHATMODEL_SERVICE = 'ollama'
    DEFAULT_CHATMODEL_KWARGS = {
        'model': 'llama3.1',
        'base_url': 'http://ollama-server:11434',
    }

    def __init__(self) -> None:
        super().__init__()
        self.chatmodel = self._create_default_chatmodel()
        self.has_system_msg_support = False

    def build(self) -> SimmpleSummarizer:
        return SimmpleSummarizer(**self.get_init_params())

    def get_init_params(self) -> dict:
        params = {
            "chatmodel": self.chatmodel,
            "has_system_msg_support": self.has_system_msg_support,
        }
        params.update(super().get_init_params())
        return params

    def set_chatmodel(self, service: str, chatmodel: BaseChatModel = None, **kwargs):
        combined_kwargs = {**self.DEFAULT_CHATMODEL_KWARGS, **kwargs}
        self.chatmodel = self._create_chatmodel(
            service=service, chatmodel=chatmodel, **combined_kwargs
        )
        return self

    def set_system_msg_support(self, has_system_msg_support: bool):
        self.has_system_msg_support = has_system_msg_support
        return self

    def _create_default_chatmodel(self) -> BaseChatModel:
        return self._create_chatmodel(
            service=self.DEFAULT_CHATMODEL_SERVICE, **self.DEFAULT_CHATMODEL_KWARGS
        )
