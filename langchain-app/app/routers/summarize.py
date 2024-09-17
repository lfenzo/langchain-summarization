from tempfile import NamedTemporaryFile

import magic
from fastapi import APIRouter, File, UploadFile

from app.models.feedback import FeedbackForm
from app.factories.store_manager_factory import StoreManagerFactory
from app.summarizers.simple_summarizer.simple_summarizer_builder import (
    SimmpleSummarizerBuilder
)
from app.summarizers.experimental.dynamic_promopts.dynamic_prompts_builder import (
    DynamicPromptSummarizerBuilder
)

router = APIRouter()

# TODO: turn this into a builder factory
SUMARIZERS = {
    'simple': SimmpleSummarizerBuilder,
    'dynamic-prompt': DynamicPromptSummarizerBuilder,
}


@router.post("/summarize/feedback")
async def upload_summary_feedback(form: FeedbackForm):
    storage_manager = StoreManagerFactory().create(manager='mongodb')
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
            SUMARIZERS['dynamic-prompt']()
            .set_loader(file_type=magic.from_buffer(contents, mime=True), file_path=tmp_file.name)
            .set_chatmodel(service='ollama', model='llama3.1')
            .set_extraction_chatmodel(service='google-genai', model='gemini-1.5-pro', temperature=0)
            .build()
        )

        return await service.summarize(file_name=file.filename)
