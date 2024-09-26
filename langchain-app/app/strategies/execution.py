import json
from typing import Any, AsyncGenerator, AsyncIterator, Dict
from abc import ABC, abstractmethod

from fastapi.responses import StreamingResponse, Response
from langchain_core.documents.base import Document
from langchain_core.messages.ai import AIMessageChunk, AIMessage
from langchain_core.runnables.base import Runnable

from app.summarizers import BaseSummarizer


class BaseExecutionStrategy(ABC):
    """
    Abstract base class for execution strategies that handle the running of a summarizer
    and the generation of summaries.

    Methods
    -------
    run(runnable, **kwargs)
        Executes a runnable task with the provided keyword arguments.
    process_summary_generation(summarizer, content, file_name)
        Asynchronously processes summary generation for the provided content.
    """

    @abstractmethod
    def run(self, runnable: Runnable, **kwargs) -> Any:
        """
        Execute a runnable with the provided keyword arguments.

        Parameters
        ----------
        runnable : Runnable
            The runnable object to execute.
        **kwargs : dict
            Additional keyword arguments to pass to the runnable.

        Returns
        -------
        Any
            The result of executing the runnable.
        """
        pass

    @abstractmethod
    async def process_summary_generation(
        self,
        summarizer: BaseSummarizer,
        content: list[Document],
        file_name: str,
    ):
        """
        Asynchronously processes the summary generation for the provided content.

        Parameters
        ----------
        summarizer : BaseSummarizer
            The summarizer instance responsible for generating the summary.
        content : list[Document]
            A list of Document objects to summarize.
        file_name : str
            The name of the file being summarized.

        Returns
        -------
        Any
            The result of the summary generation process.
        """
        pass


class StreamingStrategy(BaseExecutionStrategy):
    """
    Execution strategy for generating summaries in a streaming manner.

    Methods
    -------
    run(runnable, **kwargs)
        Executes a runnable task and returns an asynchronous iterator over message chunks.
    process_summary_generation(summarizer, content)
        Asynchronously processes summary generation and streams the result.
    """

    def run(self, runnable: Runnable, **kwargs) -> AsyncIterator[AIMessageChunk]:
        """
        Executes a runnable task and returns an asynchronous iterator over AIMessageChunks.

        Parameters
        ----------
        runnable : Runnable
            The runnable object to execute.
        **kwargs : dict
            Additional keyword arguments to pass to the runnable.

        Returns
        -------
        AsyncIterator[AIMessageChunk]
            An asynchronous iterator that yields chunks of the AI message.
        """
        return runnable.astream(**kwargs)

    async def process_summary_generation(
        self,
        summarizer: BaseSummarizer,
        content: list[Document]
    ) -> StreamingResponse:
        """
        Asynchronously processes the summary generation and streams the result.

        This method uses the summarizer to load content, generate the summary in chunks, and
        stream the summary back to the client as a JSON response.

        Parameters
        ----------
        summarizer : BaseSummarizer
            The summarizer instance responsible for generating the summary.
        content : list[Document]
            A list of Document objects to summarize.

        Returns
        -------
        StreamingResponse
            A streaming response containing chunks of the generated summary and metadata.
        """
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
                _id=summary_chunks[-1].id,
                summary=summarizer._get_summary_from_chunks(summary_chunks),
                metadata=summary_metadata,
                document=summarizer.get_original_document_as_bytes(),
            )

            yield json.dumps({"content": "", "summary_id": summary_id})

        return StreamingResponse(_create_stream_generator(), media_type='application/json')


class InvokeStrategy(BaseExecutionStrategy):
    """
    Execution strategy for generating summaries in a single invocation.

    Methods
    -------
    run(runnable, **kwargs)
        Executes a runnable task and returns the complete AI message.
    process_summary_generation(summarizer, content)
        Asynchronously processes summary generation and returns the full result.
    """

    def run(self, runnable: Runnable, **kwargs) -> AIMessage:
        """
        Executes a runnable task and returns the complete AI message.

        Parameters
        ----------
        runnable : Runnable
            The runnable object to execute.
        **kwargs : dict
            Additional keyword arguments to pass to the runnable.

        Returns
        -------
        AIMessage
            The complete AI message generated by the runnable.
        """
        return runnable.ainvoke(**kwargs)

    async def process_summary_generation(
        self,
        summarizer: BaseSummarizer,
        content: list[Document]
    ) -> Response:
        """
        Asynchronously processes the summary generation and returns the complete result.

        This method generates the entire summary at once and stores it in the system, then
        returns the summary and associated metadata.

        Parameters
        ----------
        summarizer : BaseSummarizer
            The summarizer instance responsible for generating the summary.
        content : list[Document]
            A list of Document objects to summarize.

        Returns
        -------
        Response
            A JSON response containing the generated summary and metadata.
        """
        summary = await summarizer.summarize(content=summarizer.loader.load())

        summary_metadata = summarizer.get_metadata(
            file=summarizer.get_file_path_from_loader(),
            generation_metadata=summary
        )

        summary_id = await summarizer.store_manager.store_summary(
            _id=summary.id,
            summary=summary.content,
            metadata=summary_metadata,
            document=summarizer.get_original_document_as_bytes(),
        )

        content = json.dumps({'content': summary.content, 'summary_id': summary_id})
        return Response(content=content, media_type='application/json')
