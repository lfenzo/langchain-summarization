from typing import Any, AsyncIterator, Dict

from langchain_core.documents.base import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.ai import AIMessageChunk, AIMessage
from langchain_core.runnables.base import Runnable
from langchain.prompts import ChatPromptTemplate

from app.summarizers import BaseSummarizer


class SimmpleSummarizer(BaseSummarizer):
    """
    A simple summarizer that generates document summaries using a chat model with an optional
    system message prompt. Inherits from BaseSummarizer.

    Parameters
    ----------
    chatmodel : BaseChatModel
        The chat model responsible for generating summaries.
    has_system_msg_support : bool, optional
        Indicates if the chat model supports system messages (default is False).
    **kwargs : dict
        Additional keyword arguments passed to the BaseSummarizer.
    """

    def __init__(self, chatmodel: BaseChatModel, has_system_msg_support: bool = False, **kwargs):
        """
        Initializes the SimmpleSummarizer with the specified chat model and message type support.

        Parameters
        ----------
        chatmodel : BaseChatModel
            The chat model used to generate summaries.
        has_system_msg_support : bool, optional
            Flag to indicate if the chat model supports system messages (default is False).
        **kwargs : dict
            Additional keyword arguments passed to the BaseSummarizer.
        """
        self.chatmodel = chatmodel
        self.has_system_msg_support = has_system_msg_support
        super().__init__(**kwargs)

    @property
    def prompt(self):
        msg_type = "system" if self.has_system_msg_support else "human"
        return ChatPromptTemplate.from_messages([
            (msg_type, "You are an expert multi-language AI summary writer."),
            (msg_type, "Produce a summary of the provided text."),
            (msg_type, "Do not provide an introduction, just the summary."),
            (msg_type, "The summary must contain ~30% of the length of the original"),
            (msg_type, "Summary language must be the same as the original"),
            (msg_type, "Tailor the summary to what you assume to be the document audience"),
            (msg_type, "Don't ask for follow-up questions."),
            ('human', "{text}"),
        ])

    @property
    def runnable(self, **kwargs) -> Runnable:
        return self.prompt | self.chatmodel

    def summarize(self, content: list[Document]) -> AsyncIterator[AIMessageChunk] | AIMessage:
        """
        Summarizes the provided documents by first extracting the text and then running
        the chat model to generate a summary.

        Parameters
        ----------
        content : list[Document]
            A list of documents to summarize.

        Returns
        -------
        AsyncIterator[AIMessageChunk] or AIMessage
            An asynchronous iterator over the summary chunks or the complete summary message.
        """
        text = self._get_text_from_content(content=content)
        return self.execution_strategy.run(runnable=self.runnable, input=text)

    def get_metadata(self, file: str, generation_metadata: Dict) -> Dict[str, Any]:
        """
        Generates metadata related to the summarization process, including model, prompt,
        and system message support information.

        Parameters
        ----------
        file : str
            The path or identifier of the file being summarized.
        generation_metadata : dict
            A dictionary containing metadata related to the generation process.

        Returns
        -------
        dict[str, Any]
            A dictionary containing metadata for the summarization, including information
            on the chat model, prompt, and system message support.
        """
        metadata = self._get_base_metadata(file=file, generation_metadata=generation_metadata)
        metadata.update({
            'chatmodel': repr(self.chatmodel),
            'prompt': repr(self.prompt),
            'has_system_msg_support': self.has_system_msg_support,
        })
        return metadata
