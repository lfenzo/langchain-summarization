from langchain_core.language_models.chat_models import BaseChatModel

from app.summarizers import DynamicPromptSummarizer
from app.summarizers.builders import BaseBuilder


class DynamicPromptSummarizerBuilder(BaseBuilder):
    """
    Builder class for creating a `DynamicPromptSummarizer` instance with dynamic chat model
    and extraction chat model configuration.
    """

    DEFAULT_CHATMODEL_SERVICE = 'ollama'
    DEFAULT_CHATMODEL_KWARGS = {
        'model': 'llama3.1',
        'base_url': 'http://ollama-server:11434',
    }
    DEFAULT_EXTRACTION_CHATMODEL_SERVICE = 'google-genai'
    DEFAULT_EXTRACTION_CHATMODEL_KWARGS = {
        'model': 'gemini-1.5-flash',
        'temperature': 0,
    }

    def __init__(self) -> None:
        """
        Initializes the DynamicPromptSummarizerBuilder with default chat and extraction models.

        Sets up the default instances of the chat model and extraction chat model.
        """
        super().__init__()
        self.chatmodel = self._create_default_chatmodel()
        self.extraction_chatmodel = self._create_default_extraction_chatmodel()

    def build(self) -> DynamicPromptSummarizer:
        """
        Builds and returns a `DynamicPromptSummarizer` instance.

        The summarizer is created with the configured chat model, extraction chat model, and
        other parameters inherited from the BaseBuilder.

        Returns
        -------
        DynamicPromptSummarizer
            The configured `DynamicPromptSummarizer` instance.
        """
        return DynamicPromptSummarizer(**self.get_init_params())

    def get_init_params(self) -> dict:
        """
        Retrieves the initialization parameters for building the `DynamicPromptSummarizer`.

        Includes the chat model, extraction chat model, and other parameters like loader,
        store manager, and execution strategy.

        Returns
        -------
        dict
            A dictionary of parameters needed to initialize the `DynamicPromptSummarizer`.
        """
        params = {
            "chatmodel": self.chatmodel,
            "extraction_chatmodel": self.extraction_chatmodel,
        }
        params.update(super().get_init_params())
        return params

    def set_chatmodel(self, service: str, chatmodel: BaseChatModel = None, **kwargs):
        """
        Sets the chat model, either by using an existing chat model instance or creating one.

        Combines the default chat model keyword arguments with any additional keyword arguments
        passed in.

        Parameters
        ----------
        service : str
            The name of the chat model service to use.
        chatmodel : BaseChatModel, optional
            An existing instance of `BaseChatModel`, if available (default is None).
        **kwargs : dict
            Additional keyword arguments to customize the chat model configuration.

        Returns
        -------
        DynamicPromptSummarizerBuilder
            The current instance of the builder, allowing method chaining.
        """
        combined_kwargs = {**self.DEFAULT_CHATMODEL_KWARGS, **kwargs}
        chatmodel = self._create_chatmodel(service=service, chatmodel=chatmodel, **combined_kwargs)
        self.chatmodel = chatmodel
        return self

    def set_extraction_chatmodel(self, service: str, chatmodel: BaseChatModel = None, **kwargs):
        """
        Sets the extraction chat model, either by using an existing instance or creating one.

        Combines the default extraction chat model keyword arguments with any additional keyword
        arguments passed in.

        Parameters
        ----------
        service : str
            The name of the extraction chat model service to use.
        chatmodel : BaseChatModel, optional
            An existing instance of `BaseChatModel`, if available (default is None).
        **kwargs : dict
            Additional keyword arguments to customize the extraction chat model configuration.

        Returns
        -------
        DynamicPromptSummarizerBuilder
            The current instance of the builder, allowing method chaining.
        """
        combined_kwargs = {**self.DEFAULT_EXTRACTION_CHATMODEL_KWARGS, **kwargs}
        self.extraction_chatmodel = self._create_chatmodel(
            service=service, chatmodel=chatmodel, **combined_kwargs
        )
        return self

    def _create_default_chatmodel(self) -> BaseChatModel:
        return self._create_chatmodel(
            service=self.DEFAULT_CHATMODEL_SERVICE, **self.DEFAULT_CHATMODEL_KWARGS
        )

    def _create_default_extraction_chatmodel(self) -> BaseChatModel:
        return self._create_chatmodel(
            service=self.DEFAULT_EXTRACTION_CHATMODEL_SERVICE,
            **self.DEFAULT_EXTRACTION_CHATMODEL_KWARGS
        )
