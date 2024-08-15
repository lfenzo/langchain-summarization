from abc import ABC, abstractmethod

from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables.base import Runnable
from langchain_core.stores import BaseStore, ByteStore
from langchain_core.document_loaders import BaseLoader


class BaseSummarization(ABC):
    def __init__(
        self,
        store: BaseStore,
        loader: BaseLoader,
        runnable: Runnable,
        byte_store: ByteStore,
    ) -> None:
        self.store = store
        self.loader = loader
        self.runnable = runnable
        self.byte_store = byte_store

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

    def summarize(self, file_name: str) -> str:
        content = self.loader.load()
        #doc_id = self._get_document_hash(file_name=file_name, content=content)

#        if self._is_summary_in_storage(doc_id=doc_id):
#            return "Está no storage"

        summary = self.render_summary(content)
         #self.storage.store(key=doc_id, value=summary)

        return summary

    def _get_document_hash(self, file_name: str, content: str) -> str:
        """Produce a document ID to store in cache and storage."""
        return "doc_1234"

    def _is_summary_in_storage(self, doc_id: str) -> bool:
        return True if self.store.retrieve(key=doc_id) else False
