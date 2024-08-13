from abc import ABC, abstractmethod

from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables.base import Runnable

from app.services.cache.base_cache import BaseCache
from app.services.storage.base_storage import BaseStorage


class BaseSummarization(ABC):
    def __init__(
        self,
        runnable: Runnable,
        cache: BaseCache,
        storage: BaseStorage,
    ) -> None:
        self.cache = cache
        self.storage = storage
        self.runnable = runnable

    @property
    def prompt(self):
        return ChatPromptTemplate.from_messages([
            ('system', "You produce high quality summaries in several languages"),
            ('user', "Here is the text to be summarized:\n\n{text}"),
            ('user', "You must produce the summary in the same language as the original."),
        ])

    @abstractmethod
    def render_summary(self, content: str):
        pass

    def summarize(self, filename: str, content) -> str:
        doc_id = self.get_document_hash(filename=filename, content=content)

        if self.is_summary_in_cache(doc_id=doc_id):
            return "Está no cache"
        elif self.is_summary_in_storage(doc_id=doc_id):
            return "Está no storage"

        summary = self.render_summary(content)

        self.cache.set(key=doc_id, value=summary)
        self.storage.store(key=doc_id, value=summary)

        return summary

    def get_document_hash(self, filename: str, content: str) -> str:
        """Produce a document ID to store in cache and storage."""
        return "doc_1234"

    def is_summary_in_storage(self, doc_id: str) -> bool:
        return True if self.cache.get(key=doc_id) else False

    def is_summary_in_cache(self, doc_id: str) -> bool:
        return True if self.storage.retrieve(key=doc_id) else False

