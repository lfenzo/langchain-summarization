import json
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Iterator, Dict

from fastapi.responses import StreamingResponse
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables.base import Runnable
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents.base import Document
from app.storage.base_store_manager import BaseStoreManager


class BaseSummarizer(ABC):

    def __init__(self, loader: BaseLoader, store_manager: BaseStoreManager) -> None:
        self.loader = loader
        self.store_manager = store_manager

    @property
    def prompt(self):
        return ChatPromptTemplate.from_messages([
            ('system', "You are an expert multi-language AI summary writer."),
            ('system', "Produce a summary of the provided text."),
            ('system', "Do not provide an introduction, just the summary."),
            ('system', "The summary must contain ~25% of the length of the original"),
            ('system', "Summary language must be the same as the original"),
            ('system', "Tailor the summary to what you assume to be the document audience"),
            ('system', "Don't ask for follouw-up questions."),
            ('human', "{text}"),
        ])

    @property
    def runnable(self) -> Runnable:
        return self.create_runnable()

    @abstractmethod
    def create_runnable(self, **kwargs) -> Runnable:
        pass

    @abstractmethod
    def get_metadata(self, file_name: str, content) -> dict[str, Any]:
        pass

    def render_summary(self, content) -> Iterator:
        return self.runnable.astream(input=self._get_content_text(content=content))

    async def summarize(self, file_name: str) -> StreamingResponse:
        content = self.loader.load()
        summary_chunks = []

        async def stream_summary() -> AsyncGenerator[Dict[str, Any], None]:
            async for chunk in self.render_summary(content):
                chunk_data = {"summary_chunk": chunk, "document_id": None}
                summary_chunks.append(chunk_data)
                yield json.dumps(chunk_data)

            summary = self._get_summary_from_chunks(summary_chunks)
            metadata = self.get_metadata(file_name=file_name, content=summary)
            document_bytes = self._get_document_bytes()

            summary_id = await self.store_manager.store_summary(
                summary=summary,
                metadata=metadata,
                document=document_bytes,
            )

            yield json.dumps({"summary_chunk": "", "summary_id": summary_id})

        return StreamingResponse(stream_summary(), media_type="application/json")

    def _get_content_text(self, content: list[Document]) -> str:
        return "".join([page.page_content + "\n" for page in content])

    def _get_summary_from_chunks(self, summary_chunks: dict[str, str]) -> str:
        return "".join([chunk['summary_chunk'] for chunk in summary_chunks])

    def _get_document_bytes(self) -> bytes:
        has_blob_perser = hasattr(self.loader, 'blob_parser')
        path = self.loader.blob_loader.path if has_blob_perser else self.loader.file_path
        with open(path, 'rb') as file:
            return file.read()
