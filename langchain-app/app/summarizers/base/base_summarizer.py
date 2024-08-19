from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables.base import Runnable
from langchain_core.document_loaders import BaseLoader

from app.storage.base_store_manager import BaseStoreManager


class BaseSummarizer(ABC):

    def __init__(
        self,
        loader: BaseLoader,
        runnable: Runnable,
        store_manager: BaseStoreManager,
    ) -> None:
        self.loader = loader
        self.runnable = runnable
        self.store_manager = store_manager

    @property
    def prompt(self):
        return ChatPromptTemplate.from_messages([
            ('system', "You produce high quality summaries in several languages"),
            ('user', "You must produce the summary in the same language as the original."),
            ('user', "Here is the text to be summarized:\n\n{text}"),
        ])

    @abstractmethod
    def render_summary(self, content):
        pass

    @abstractmethod
    def get_metadata(self, file_name: str, content: AIMessage) -> dict[str, Any]:
        pass

    def get_document_bytes(self) -> bytes:
        with open(self.loader.file_path, 'rb') as file:
            return file.read()

    def summarize(self, file_name: str) -> str:
        content = self.loader.load()
        summary = self.render_summary(content)
        metadata = self.get_metadata(file_name=file_name, content=summary)
        document_bytes = self.get_document_bytes()

        self.store_manager.store_summary(
            metadata=metadata,
            summary=summary.content,
            document=document_bytes,
        )
        return summary
