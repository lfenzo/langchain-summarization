from abc import ABC, abstractmethod

from app.models.feedback import FeedbackForm


class BaseStoreManager(ABC):
    """
    Abstract base class that defines the interface for storing summaries and related metadata
    in a database, as well as handling summary feedback.

    Methods
    -------
    get_summary()
        Abstract method to retrieve a summary from the database.
    store_summary(_id, summary, metadata, document)
        Abstract method to store a summary and its associated metadata in the database.
    store_summary_feedback(form)
        Abstract method to store user feedback on the generated summary.
    """

    @abstractmethod
    def get_summary(self):
        """
        Retrieve a summary from the database.

        Returns
        -------
        Any
            The retrieved summary from the database.
        """
        pass

    @abstractmethod
    def store_summary(self, _id: str, summary: str, metadata: dict, document: bytes) -> str:
        """
        Store a summary and its related metadata in the database.

        This method saves the generated summary produced by the LLM, along with its associated
        metadata and the original document in byte form. The `_id` parameter is the identifier
        returned by the BaseModel execution, encapsulated within an `AIMessage` object, which
        results from a model invocation.

        Parameters
        ----------
        _id : str
            The unique identifier returned by the BaseModel execution, encapsulated in an
            `AIMessage` object.
        summary : str
            The summary generated by the language model (LLM).
        metadata : dict
            A dictionary containing information about the summary generation, including metadata
            about the original document, class, generation metadata, and other relevant details.
        document : bytes
            The original document in byte format (e.g. a PDF, audio, or other file).

        Returns
        -------
        str
            The ID of the stored document (typically the same as the passed `_id`).
        """
        ...

    @abstractmethod
    def store_summary_feedback(self, form: FeedbackForm):
        """
        Store feedback for a generated summary in the database.

        Parameters
        ----------
        form : FeedbackForm
            A form containing the user's feedback on the generated summary, including ratings
            or comments about the quality of the summarization.

        Returns
        -------
        Any
            A confirmation of feedback storage, typically a success message or status.
        """
        pass
