from langchain_core.document_loaders import BaseLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.parsers.audio import FasterWhisperParser


class LoaderFactory:

    def __init__(self):
        self.loader_from_mime_type = {
            'application/pdf': self._get_pdf_loader,
            'video/mp4': self._get_audio_loader,
        }

    def create(self, file_type: str, file_path: str, **kwargs) -> BaseLoader:
        return self.loader_from_mime_type[file_type](file_path=file_path, **kwargs)

    def _get_pdf_loader(self, file_path: str, **kwargs):
        return PyMuPDFLoader(file_path=file_path, **kwargs)

    def _get_audio_loader(self, file_path: str, model_size: str = 'large-v3'):
        return GenericLoader.from_filesystem(
            path=file_path,
            parser=FasterWhisperParser(model_size=model_size),
        )
