from langchain_core.language_models.chat_models import BaseChatModel

from app.summarizers import SimmpleSummarizer
from app.summarizers.builders import BaseBuilder


class SimmpleSummarizerBuilder(BaseBuilder):
    """
    Builder class for creating a `SimmpleSummarizer` instance with configurable chat model
    and system message support.
    """

    DEFAULT_CHATMODEL_SERVICE = 'ollama'
    DEFAULT_CHATMODEL_KWARGS = {
        'model': 'llama3.1',
        'base_url': 'http://ollama-server:11434',
    }

    def __init__(self) -> None:
        """
        Initializes the SimmpleSummarizerBuilder with the default chat model.

        Sets up the default chat model and initializes the `has_system_msg_support` flag.
        """
        super().__init__()
        self.chatmodel = self._create_default_chatmodel()
        self.has_system_msg_support = False

    def build(self) -> SimmpleSummarizer:
        """
        Builds and returns a `SimmpleSummarizer` instance.

        The summarizer is created with the configured chat model, system message support,
        and other parameters inherited from the BaseBuilder.

        Returns
        -------
        SimmpleSummarizer
            The configured `SimmpleSummarizer` instance.
        """
        return SimmpleSummarizer(**self.get_init_params())

    def get_init_params(self) -> dict:
        """
        Retrieves the initialization parameters for building the `SimmpleSummarizer`.

        Includes the chat model, system message support, and other parameters like loader,
        store manager, and execution strategy.

        Returns
        -------
        dict
            A dictionary of parameters needed to initialize the `SimmpleSummarizer`.
        """
        params = {
            "chatmodel": self.chatmodel,
            "has_system_msg_support": self.has_system_msg_support,
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
        SimmpleSummarizerBuilder
            The current instance of the builder, allowing method chaining.
        """
        combined_kwargs = {**self.DEFAULT_CHATMODEL_KWARGS, **kwargs}
        self.chatmodel = self._create_chatmodel(
            service=service, chatmodel=chatmodel, **combined_kwargs
        )
        return self

    def set_system_msg_support(self, has_system_msg_support: bool):
        """
        Configures the system message support for the summarizer.

        Parameters
        ----------
        has_system_msg_support : bool
            A flag indicating whether the chat model supports system messages.

        Returns
        -------
        SimmpleSummarizerBuilder
            The current instance of the builder, allowing method chaining.
        """
        self.has_system_msg_support = has_system_msg_support
        return self

    def _create_default_chatmodel(self) -> BaseChatModel:
        return self._create_chatmodel(
            service=self.DEFAULT_CHATMODEL_SERVICE, **self.DEFAULT_CHATMODEL_KWARGS
        )
