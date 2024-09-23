from langchain_core.document_loaders import BaseLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import RemoteFasterWhisperParser


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

    def _get_audio_loader(self, file_path: str):
        return GenericLoader.from_filesystem(
            path=file_path,
            parser=RemoteFasterWhisperParser(
                base_url='http://faster-whisper-server:9000',
                model_size='large-v3',
            ),
        )
