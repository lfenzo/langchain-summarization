import magic
from io import BytesIO
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, UploadFile, HTTPException
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

from app.services.summarization.ollama_summarization import OllamaSummarization
from app.services.cache.dummy import DummyCache
from app.services.storage.dummy import DummyStorage

router = APIRouter()


def get_document_format(document_name: str):
    return document_name.split('.')[-1].lower()


def load_document(file_type: str, file_path: str) -> list[Document]:
    """Loads a document based on the file type."""
    if file_type == "application/pdf":
        loader = PyMuPDFLoader(file_path=file_path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    return loader.load()


@router.post("/test-upload")
async def test_upload_document(file: UploadFile = File(...)):
    contents = await file.read()
    print(f"Received file: {file.filename}, size: {len(contents)} bytes")
    return {"filename": file.filename, "size": len(contents)}


@router.post("/summarize")
async def summarize(file: UploadFile = File(...)):
    contents = await file.read()
    print(f"Received file: {file.filename}, size: {len(contents)} bytes")

    with NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(contents)
        tmp_file.flush()
        tmp_file.seek(0)  # Ensure the file pointer is at the start

        file_type = magic.from_buffer(contents, mime=True)
        document = load_document(file_type=file_type, file_path=tmp_file.name)

        service = OllamaSummarization(
            cache=DummyCache(),
            storage=DummyStorage(),
            base_url="http://ollama-server:11434",
        )

        return service.summarize(filename=file.filename, content=document)
