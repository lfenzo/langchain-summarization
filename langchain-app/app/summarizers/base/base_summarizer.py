import json
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, AsyncIterator, Dict

from langchain_core.messages.ai import AIMessageChunk
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
    def get_metadata(self, file_name: str, content, last_chunk: dict) -> dict[str, Any]:
        pass

    def create_runnable(self, **kwargs) -> Runnable:
        return self.prompt | self.model

    def render_summary(self, content) -> AsyncIterator[AIMessageChunk]:
        return self.runnable.astream(input=self._get_text_from_content(content=content))

    async def summarize(self, file_name: str) -> StreamingResponse:
        content_to_summarize = self.loader.load()
        summary_chunks = []

        async def stream_summary():
            async for chunk in self.render_summary(content=content_to_summarize):
                summary_chunks.append(chunk)
                yield json.dumps({"content": chunk.content})

            summary = self._get_summary_from_chunks(summary_chunks)
            original_document_in_bytes = self._get_document_bytes()
            metadata = self.get_metadata(
                content=summary,
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
