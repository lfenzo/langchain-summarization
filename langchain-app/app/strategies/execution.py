import json
from typing import Any, AsyncGenerator, AsyncIterator, Dict
from abc import ABC, abstractmethod

from fastapi.responses import StreamingResponse, Response
from langchain_core.documents.base import Document
from langchain_core.messages.ai import AIMessageChunk, AIMessage
from langchain_core.runnables.base import Runnable

from app.summarizers import BaseSummarizer


class BaseExecutionStrategy(ABC):

    @abstractmethod
    def run(self, runnable: Runnable, **kwargs) -> Any:
        """Execute a runnable with **kwargs as inputs"""
        ...

    @abstractmethod
    async def process_summary_generation(
        self,
        summarizer: BaseSummarizer,
        content: list[Document],
        file_name: str,
    ):
        ...


class StreamingStrategy(BaseExecutionStrategy):

    def run(self, runnable: Runnable, **kwargs) -> AsyncIterator[AIMessageChunk]:
        return runnable.astream(**kwargs)

    async def process_summary_generation(self, summarizer: BaseSummarizer, content: list[Document]):
        content_to_summarize = summarizer.loader.load()
        summary_chunks = []

        async def _create_stream_generator() -> AsyncGenerator[Dict[str, Any], None]:
            async for chunk in summarizer.summarize(content=content_to_summarize):
                summary_chunks.append(chunk)
                yield json.dumps({"content": chunk.content})

            summary_metadata = summarizer.get_metadata(
                file=summarizer.get_file_path_from_loader(),
                generation_metadata=summary_chunks[-1]
            )

            summary_id = await summarizer.store_manager.store_summary(
                summary=summarizer._get_summary_from_chunks(summary_chunks),
                metadata=summary_metadata,
                document=summarizer.get_original_document_as_bytes(),
            )

            yield json.dumps({"content": "", "summary_id": summary_id})

        return StreamingResponse(_create_stream_generator(), media_type='application/json')


class InvokeStrategy(BaseExecutionStrategy):

    def run(self, runnable: Runnable, **kwargs) -> AIMessage:
        return runnable.ainvoke(**kwargs)

    async def process_summary_generation(self, summarizer: BaseSummarizer, content: list[Document]):
        summary = await summarizer.summarize(content=summarizer.loader.load())

        summary_metadata = summarizer.get_metadata(
            file=summarizer.get_file_path_from_loader(),
            generation_metadata=summary
        )

        print(summary)

        summary_id = await summarizer.store_manager.store_summary(
            summary=summary.content,
            metadata=summary_metadata,
            document=summarizer.get_original_document_as_bytes(),
        )

        content = json.dumps({'content': summary.content, 'summary_id': summary_id})
        return Response(content=content, media_type='application/json')
