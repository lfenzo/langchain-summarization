from tempfile import NamedTemporaryFile

import magic
from fastapi import APIRouter, File, UploadFile

from app.summarizers.ollama.ollama_builder import (
    OllamaSummarizationBuilder
)
from app.summarizers.experimental.reader_centered.reader_centered_builder import (
    ReaderCenteredSummarizationBuilder
)
from app.summarizers.experimental.document_centered.document_centered_builder import (
    DocumentCenteredSummarizationBuilder
)

from app.summarizers.google_vertexai.google_vertexai_builder import (
    GoogleVertexAISummarizerBuilder
)

router = APIRouter()

SUMARIZERS = {
    'ollama': OllamaSummarizationBuilder,
    'google': GoogleVertexAISummarizerBuilder,
    'reader_info_extraction': ReaderCenteredSummarizationBuilder,
    'document_info_extraction': DocumentCenteredSummarizationBuilder,
}


@router.post("/test-upload")
async def test_upload_document(file: UploadFile = File(...)):
    contents = await file.read()
    print(f"Received file: {file.filename}, size: {len(contents)} bytes")
    return {"filename": file.filename, "size": len(contents)}


@router.post("/summarize")
async def summarize(file: UploadFile = File(...)):
    contents = await file.read()

    with NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(contents)
        tmp_file.flush()
        tmp_file.seek(0)  # Ensure the file pointer is at the start

        service = (
            SUMARIZERS['google']()
            .set_loader(file_type=magic.from_buffer(contents, mime=True), file_path=tmp_file.name)
            .build()
        )

        return await service.summarize(file_name=file.filename)
