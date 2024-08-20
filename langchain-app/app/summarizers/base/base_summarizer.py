from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator

from fastapi.responses import StreamingResponse
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
    def get_metadata(self, file_name: str, content) -> dict[str, Any]:
        pass

    def get_document_bytes(self) -> bytes:
        with open(self.loader.file_path, 'rb') as file:
            return file.read()

    async def summarize(self, file_name: str) -> str:
        content = self.loader.load()
        summary_parts = []

        async def stream_summary() -> AsyncGenerator[str, None]:
            async for chunk in self.render_summary(content):
                summary_parts.append(chunk)
                yield chunk

            full_summary = "".join(summary_parts)
            document_bytes = self.get_document_bytes()
            metadata = self.get_metadata(file_name=file_name, content=full_summary)

            await self.store_manager.store_summary(
                metadata=metadata,
                summary=full_summary,
                document=document_bytes,
            )

        return StreamingResponse(stream_summary(), media_type="text/plain")
