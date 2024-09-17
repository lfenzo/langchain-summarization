import json
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, AsyncIterator, Dict

from fastapi.responses import StreamingResponse
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents.base import Document
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.runnables.base import Runnable

from app.storage import BaseStoreManager


class BaseSummarizer(ABC):

    def __init__(self, loader: BaseLoader, store_manager: BaseStoreManager) -> None:
        self.loader = loader
        self.store_manager = store_manager

    @property
    def runnable(self) -> Runnable:
        ...

    @abstractmethod
    def get_metadata(self, file_name: str, last_chunk: dict) -> dict[str, Any]:
        ...

    @abstractmethod
    def render_summary(self, content: list[Document]) -> AsyncIterator[AIMessageChunk]:
        ...

    async def summarize(self, file_name: str) -> StreamingResponse:
        content_to_summarize = self.loader.load()
        summary_chunks = []

        async def stream_summary() -> AsyncGenerator[Dict[str, Any], None]:
            async for chunk in self.render_summary(content=content_to_summarize):
                summary_chunks.append(chunk)
                yield json.dumps({"content": chunk.content})

            summary = self._get_summary_from_chunks(summary_chunks)
            original_document_in_bytes = self._get_document_bytes()
            metadata = self.get_metadata(
                file_name=file_name,
                last_chunk=summary_chunks[-1],
            )

            summary_id = await self.store_manager.store_summary(
                summary=summary,
                metadata=metadata,
                document=original_document_in_bytes,
            )

            yield json.dumps({"content": "", "summary_id": summary_id})

        return StreamingResponse(stream_summary(), media_type="application/json")

    def _get_text_from_content(self, content: list[Document]) -> str:
        return "".join([page.page_content + "\n" for page in content])

    def _get_summary_from_chunks(self, summary_chunks: list[AIMessageChunk]) -> str:
        return "".join([chunk.content for chunk in summary_chunks])

    def _get_document_bytes(self) -> bytes:
        has_blob_perser = hasattr(self.loader, 'blob_parser')
        path = self.loader.blob_loader.path if has_blob_perser else self.loader.file_path
        with open(path, 'rb') as file:
            return file.read()

    def _get_base_metadata(self, file_name: str, last_chunk: Dict) -> Dict[str, Any]:
        return {
            'input_file_name': file_name,
            'summarizer': self.__class__.__name__,
            'loader': repr(self.loader),
            **last_chunk.response_metadata,
            **last_chunk.usage_metadata,
        }
