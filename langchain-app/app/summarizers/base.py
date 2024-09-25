from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents.base import Document
from langchain_core.messages.ai import AIMessageChunk, AIMessage
from langchain_core.runnables.base import Runnable

from app.storage import BaseStoreManager


class BaseSummarizer(ABC):

    def __init__(
        self,
        loader: BaseLoader,
        store_manager: BaseStoreManager,
        execution_strategy: "BaseExecutionStrategy",
    ) -> None:
        self.loader = loader
        self.store_manager = store_manager
        self.execution_strategy = execution_strategy

    @property
    def runnable(self) -> Runnable:
        ...

    @abstractmethod
    def get_metadata(self, file_name: str, last_chunk: dict) -> dict[str, Any]:
        ...

    @abstractmethod
    def summarize(self, content: list[Document]) -> AsyncIterator[AIMessageChunk] | AIMessage:
        ...

    async def process_summary_generation(self, file_name: str):
        return await self.execution_strategy.process_summary_generation(
            summarizer=self,
            content=self.loader.load(),
            file_name=file_name,
        )

    def get_original_document_as_bytes(self) -> bytes:
        has_blob_perser = hasattr(self.loader, 'blob_parser')
        path = self.loader.blob_loader.path if has_blob_perser else self.loader.file_path
        with open(path, 'rb') as file:
            return file.read()

    def _get_text_from_content(self, content: list[Document]) -> str:
        return "".join([page.page_content + "\n" for page in content])

    def _get_summary_from_chunks(self, summary_chunks: list[AIMessageChunk]) -> str:
        return "".join([chunk.content for chunk in summary_chunks])

    def _get_base_metadata(self, file_name: str, generation_metadata: Dict) -> Dict[str, Any]:
        return {
            'input_file_name': file_name,
            'summarizer': self.__class__.__name__,
            'loader': repr(self.loader),
            **generation_metadata.response_metadata,
            **generation_metadata.usage_metadata,
        }
