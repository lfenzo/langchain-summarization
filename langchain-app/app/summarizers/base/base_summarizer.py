from abc import ABC, abstractmethod

from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables.base import Runnable
from langchain_core.stores import BaseStore, ByteStore
from langchain_core.document_loaders import BaseLoader


class BaseSummarizer(ABC):

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
        return self.render_summary(content)
