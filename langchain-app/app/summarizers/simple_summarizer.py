from typing import Any, AsyncIterator, Dict

from langchain_core.documents.base import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.runnables.base import Runnable
from langchain.prompts import ChatPromptTemplate

from app.summarizers import BaseSummarizer


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

    @property
    def runnable(self, **kwargs) -> Runnable:
        return self.prompt | self.chatmodel

    def render_summary(self, content: list[Document]) -> AsyncIterator[AIMessageChunk]:
        return self.runnable.astream(input=self._get_text_from_content(content=content))

    def get_metadata(self, file_name: str, last_chunk: Dict) -> Dict[str, Any]:
        metadata = self._get_base_metadata(file_name=file_name, last_chunk=last_chunk)
        metadata.update({
            'chatmodel': repr(self.chatmodel),
            'prompt': repr(self.prompt),
            'has_system_msg_support': self.has_system_msg_support,
        })
        return metadata
