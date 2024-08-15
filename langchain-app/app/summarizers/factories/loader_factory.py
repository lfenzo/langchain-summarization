from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.document_loaders import BaseLoader


class LoaderFactory:

    def __init__(self):
        self.loader_from_mime_type = {
            'application/pdf': PyMuPDFLoader,
        }

    def create(self, file_type: str, file_path: str, **kwargs) -> BaseLoader:
        return self.loader_from_mime_type[file_type](file_path=file_path, **kwargs)
