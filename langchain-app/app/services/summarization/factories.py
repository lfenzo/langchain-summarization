from langchain_community.cache import InMemoryCache
from langchain_core.document_loaders import BaseLoader
from langchain_community.document_loaders import PyMuPDFLoader

from app.services.summarization.ollama_summarization import OllamaSummarization


class FileLoaderFactory:

    def __init__(self):
        self.loader_from_mime_type = {
            'application/pdf': PyMuPDFLoader,
        }

    def create(self, file_type: str, file_path: str, **kwargs) -> BaseLoader:
        return self.loader_from_mime_type[file_type](file_path=file_path, **kwargs)


class SummarizationFactory:

    def __init__(self):
        self.loader_factory = FileLoaderFactory()
        self.summarizer_from_method = {
            'ollama': self._create_ollama_summarization_service
        }

    def create(self, summarizer: str, file_type: str, file_path: str, **kwargs):
        loader = self.loader_factory.create(
            file_path=file_path,
            file_type=file_type,
            **kwargs
        )
        return self.summarizer_from_method[summarizer](loader=loader, **kwargs)

    def _create_ollama_summarization_service(self, loader, **kwargs):
        return OllamaSummarization(
            model='llama3.1',
            base_url='http://ollama-server:11434',
            cache=InMemoryCache(),
            loader=loader,
            store=None,
            byte_store=None,
            **kwargs,
        )
