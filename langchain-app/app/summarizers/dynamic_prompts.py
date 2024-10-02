from typing import Any, AsyncIterator, Dict

from langchain_core.documents.base import Document
from langchain_core.messages.ai import AIMessageChunk, AIMessage
from langchain.chat_models.base import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from app.models import DocumentInfo
from app.summarizers import BaseSummarizer


class DynamicPromptSummarizer(BaseSummarizer):
    """
    Summarizer that dynamically generates prompts for both extraction and summarization
    using a chat-based model. This class builds on the BaseSummarizer and uses language
    models to extract structured data and summarize documents.

    Parameters
    ----------
    chatmodel : BaseChatModel
        The main chat model used for generating summaries.
    extraction_chatmodel : BaseChatModel
        The chat model used for extracting structured information from the documents.
    **kwargs : dict
        Additional keyword arguments passed to the BaseSummarizer.
    """

    def __init__(
        self,
        chatmodel: BaseChatModel,
        extraction_chatmodel: BaseChatModel,
        **kwargs,
    ) -> None:
        """
        Initialize the DynamicPromptSummarizer with chat models for summarization and extraction.

        Parameters
        ----------
        chatmodel : BaseChatModel
            The main chat model used for summarization.
        extraction_chatmodel : BaseChatModel
            The chat model used for extracting structured information from documents.
        **kwargs : dict
            Additional keyword arguments passed to the BaseSummarizer.
        """
        super().__init__(**kwargs)
        self.chatmodel = chatmodel
        self.extraction_chatmodel = extraction_chatmodel

    @property
    def extraction_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            (
                "human",
                """
                You are an expert extraction algorithm, specialized in extracting structured
                information. Your task is to accurately identify and extract relevant attributes
                from the provided text. For each attribute, if the value cannot be determined from
                the text, return 'null' as the attribute's value. Ensure the extracted information
                is concise, relevant, and structured according to the required format."
                """
            ),
            ("human", "{text}"),
        ])

    @property
    def summarization_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            (
                "human",
                """
                You are an advanced AI specializing in identifying and summarizing the most
                important and relevant information from complex documents. Your task is to create
                a detailed summary by focusing on the key ideas, core arguments, and supporting
                details.
                """
            ),
            (
                "human",
                """
                Ensure the summary is approximately 30% of the original length. Prioritize the
                following:
                - Major themes and critical points
                - Important supporting details that enhance the key points
                - Exclude redundant or trivial information
                """
            ),
            (
                "human",
                """
                Follow these guidelines when generating the summary:
                - Text Type: {text_type} (e.g., report, presentation, article)
                - Media Type: {media_type} (e.g., PDF, PPT, DOC)
                - Domain: {document_domain} (e.g., finance, medical, legal)
                - Audience: {audience} (e.g., general, experts)
                - Audience Expertise: {audience_expertise} (e.g., beginner, intermediate, advanced)
                - Focus on Document Key Points: {key_points}
                """
            ),
            (
                "human",
                """
                The summary must be written in the same language as the input document, maintaining
                the document’s formal tone and style. Avoid introductory phrases and external
                knowledge. Simply focus on what is present in the document itself.
                """
            ),
            ("human", "{text}"),
        ])

    @property
    def extraction_chain(self):
        return (
            self.extraction_prompt
            | self.extraction_chatmodel.with_structured_output(schema=DocumentInfo)
        )

    @property
    def summarization_chain(self):
        return self.summarization_prompt | self.chatmodel

    def summarize(self, content: list[Document]) -> AsyncIterator[AIMessageChunk] | AIMessage:
        """
        Summarizes the provided documents after extracting structured information.

        The text content is first extracted from the documents, then structured information
        is extracted using the extraction chain. Finally, the summarization chain is invoked
        to generate the final summary.

        Parameters
        ----------
        content : list[Document]
            A list of documents to summarize.

        Returns
        -------
        AsyncIterator[AIMessageChunk] or AIMessage
            An asynchronous iterator over message chunks or a complete AI message.
        """
        text = self._get_text_from_content(content=content)
        structured_information = self.extraction_chain.invoke({"text": text})
        return self.execution_strategy.run(
            runnable=self.summarization_chain,
            kwargs={"text": text, **structured_information.dict()},
        )

    def get_metadata(self, file: str, generation_metadata: Dict) -> Dict[str, Any]:
        """
        Generates metadata for the summarization process, including model and prompt information.

        This method builds on the base metadata by adding details specific to the summarization
        and extraction chat models and prompts.

        Parameters
        ----------
        file : str
            The path or identifier of the file being summarized.
        generation_metadata : dict
            A dictionary containing metadata related to the summarization process.

        Returns
        -------
        dict[str, Any]
            A dictionary containing metadata for the summarization, including information
            on the chat models, prompts, and extraction schema.
        """
        metadata = self._get_base_metadata(file=file, generation_metadata=generation_metadata)
        metadata.update({
            'chatmodel': repr(self.chatmodel),
            "summarization_prompt": repr(self.summarization_prompt),
            'extraction_chatmodel': repr(self.extraction_chatmodel),
            "extraction_prompt": repr(self.extraction_prompt),
            "structured_straction_schema": DocumentInfo.__class__.__name__,
        })
        return metadata
