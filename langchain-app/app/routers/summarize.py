from tempfile import NamedTemporaryFile

import magic
from fastapi import APIRouter, File, UploadFile

from app.models.feedback import FeedbackForm
from app.factories.store_manager_factory import StorageManagerFactory
from app.summarizers.ollama.ollama_builder import OllamaSummarizationBuilder
from app.summarizers.google_genai.google_genai_builder import GoogleGenAISummarizerBuilder
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

# TODO: turn this into a builder factory
SUMARIZERS = {
    'ollama': OllamaSummarizationBuilder,
    'google-vertex': GoogleVertexAISummarizerBuilder,
    'google-genai': GoogleGenAISummarizerBuilder,
    'reader_info_extraction': ReaderCenteredSummarizationBuilder,
    'document_info_extraction': DocumentCenteredSummarizationBuilder,
}


@router.post("/summarize/feedback")
async def upload_summary_feedback(form: FeedbackForm):
    storage_manager = StorageManagerFactory().create(manager='mongodb')
    await storage_manager.store_summary_feedback(form=form)
    return {
        'user': form.user,
        'document_id': form.document_id,
    }


@router.post("/summarize")
async def summarize(file: UploadFile = File(...)):
    contents = await file.read()

    with NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(contents)
        tmp_file.flush()
        tmp_file.seek(0)  # Ensure the file pointer is at the start

        service = (
            SUMARIZERS['ollama']()
            .set_loader(file_type=magic.from_buffer(contents, mime=True), file_path=tmp_file.name)
            .set_model_name('llama3.1')
            .build()
        )

        return await service.summarize(file_name=file.filename)
