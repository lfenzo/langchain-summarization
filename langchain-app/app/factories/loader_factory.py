from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)


class LoaderFactory:
    def create_loader(self, mime_type: str, **kwargs):
        if mime_type == "application/pdf":
            return PyMuPDFLoader()
