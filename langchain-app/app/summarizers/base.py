from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict

from fastapi.responses import StreamingResponse, Response
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents.base import Document
from langchain_core.messages.ai import AIMessageChunk, AIMessage

from app.storage import BaseStoreManager


class BaseSummarizer(ABC):
    """
    Abstract base class for summarizers that process documents and generate summaries.

    Parameters
    ----------
    loader : BaseLoader
        The document loader used to retrieve content for summarization.
    store_manager : BaseStoreManager
        Instance of the store manager handling storage-related operations.
    execution_strategy : BaseExecutionStrategy
        Strategy for executing the summarization process.
    """

    def __init__(
        self,
        loader: BaseLoader,
        store_manager: BaseStoreManager,
        execution_strategy: "BaseExecutionStrategy",
    ) -> None:
        """
        Initialize the BaseSummarizer with a loader, store manager, and execution strategy.

        Parameters
        ----------
        loader : BaseLoader
            The document loader responsible for loading content for summarization.
        store_manager : BaseStoreManager
            The store manager responsible for managing summary storage and retrieval.
        execution_strategy : BaseExecutionStrategy
            Defines the strategy to be used for executing the summarization process.
        """
        self.loader = loader
        self.store_manager = store_manager
        self.execution_strategy = execution_strategy

    @abstractmethod
    def get_metadata(self, file: str, generation_metadata: dict) -> dict[str, Any]:
        """
        Abstract method to extract metadata for the specified file.

        Parameters
        ----------
        file : str
            The path or identifier of the file for which metadata is being retrieved.
        generation_metadata : dict
            Metadata related to the generation process.

        Returns
        -------
        dict[str, Any]
            A dictionary containing metadata information for the file.
        """
        pass

    @abstractmethod
    def summarize(self, content: list[Document]) -> AsyncIterator[AIMessageChunk] | AIMessage:
        """
        Abstract method to generate a summary from a list of documents.

        Parameters
        ----------
        content : list[Document]
            A list of documents to be summarized.

        Returns
        -------
        AsyncIterator[AIMessageChunk] or AIMessage
            An asynchronous iterator over chunks of the summary or the complete AI message.
        """
        pass

    async def process_summary_generation(self) -> Response | StreamingResponse:
        """
        Asynchronously processes the generation of a summary using the execution strategy.

        This method loads the content from the loader and then invokes the
        execution strategy to handle the summarization process.

        Returns
        -------
        Response or StreamingResponse
            A FastAPI response or streaming response object containing the summary.
        """
        return await self.execution_strategy.process_summary_generation(
            summarizer=self,
            content=self.loader.load(),
        )

    def get_original_document_as_bytes(self) -> bytes:
        """
        Retrieves the original document as bytes from the file path provided by the loader.

        Returns
        -------
        bytes
            The contents of the original document in bytes.
        """
        with open(self.get_file_path_from_loader(), 'rb') as file:
            return file.read()

    def get_file_path_from_loader(self) -> str:
        """
        Retrieves the file path of the document from the loader.

        Returns
        -------
        str
            The file path of the document loaded by the loader.
        """
        has_blob_perser = hasattr(self.loader, 'blob_parser')
        return self.loader.blob_loader.path if has_blob_perser else self.loader.file_path

    def _get_text_from_content(self, content: list[Document]) -> str:
        """
        Extracts and concatenates text from a list of Document objects into a single string.

        Parameters
        ----------
        content : list[Document]
            A list of Document objects to extract and combine the text from.

        Returns
        -------
        str
            The concatenated text from the provided documents.
        """
        return "".join([page.page_content + "\n" for page in content])

    def _get_summary_from_chunks(self, summary_chunks: list[AIMessageChunk]) -> str:
        """
        Concatenates content from a list of AIMessageChunk objects to form a complete summary.

        Parameters
        ----------
        summary_chunks : list[AIMessageChunk]
            A list of message chunks representing parts of the summary.

        Returns
        -------
        str
            The concatenated summary string from the message chunks.
        """
        return "".join([chunk.content for chunk in summary_chunks])

    def _get_base_metadata(self, file: str, generation_metadata: Dict) -> Dict[str, Any]:
        """
        Constructs the base metadata for a file, including summarizer and loader information.

        Parameters
        ----------
        file : str
            The path or identifier of the file being summarized.
        generation_metadata : dict
            A dictionary containing metadata related to the generation process.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the base metadata for the summarization process.
        """
        return {
            'input_file': file,
            'summarizer': self.__class__.__name__,
            'loader': repr(self.loader),
            **generation_metadata.response_metadata,
            **generation_metadata.usage_metadata,
        }
