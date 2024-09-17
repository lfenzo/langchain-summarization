from typing import Any

from langchain_core.runnables.base import Runnable
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import ChatPromptTemplate

from app.summarizers.base.base_summarizer import BaseSummarizer


class SimmpleSummarizer(BaseSummarizer):

    def __init__(self, chatmodel: BaseChatModel, has_system_msg_support: bool = False, **kwargs):
        self.chatmodel = chatmodel
        self.has_system_msg_support = has_system_msg_support
        super().__init__(**kwargs)

    @property
    def prompt(self):
        msg_type = "system" if self.has_system_msg_support else "human"
        return ChatPromptTemplate.from_messages([
            (msg_type, "You are an expert multi-language AI summary writer."),
            (msg_type, "Produce a summary of the provided text."),
            (msg_type, "Do not provide an introduction, just the summary."),
            (msg_type, "The summary must contain ~25% of the length of the original"),
            (msg_type, "Summary language must be the same as the original"),
            (msg_type, "Tailor the summary to what you assume to be the document audience"),
            (msg_type, "Don't ask for follouw-up questions."),
            ('human', "{text}"),
        ])

    def create_runnable(self, **kwargs) -> Runnable:
        return self.prompt | self.chatmodel

    def get_metadata(self, file_name: str, last_chunk: dict) -> dict[str, Any]:
        return {
            'input_file_name': file_name,
            'summarizer': self.__class__.__name__,
            'loader': repr(self.loader),
            'chatmodel': repr(self.chatmodel),
            'prompt': repr(self.prompt),
            'has_system_msg_support': self.has_system_msg_support,
            **last_chunk.response_metadata,
            **last_chunk.usage_metadata,
        }
