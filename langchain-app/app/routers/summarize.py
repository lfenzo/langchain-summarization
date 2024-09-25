from tempfile import NamedTemporaryFile

import magic
from fastapi import APIRouter, File, UploadFile

from app.models import FeedbackForm
from app.factories import StoreManagerFactory
from app.summarizers.builders import SimmpleSummarizerBuilder, DynamicPromptSummarizerBuilder


router = APIRouter()

SUMARIZERS = {
    'simple': SimmpleSummarizerBuilder,
    'dynamic-prompt': DynamicPromptSummarizerBuilder,
}


@router.post("/summarize/feedback")
async def upload_summary_feedback(form: FeedbackForm):
    storage_manager = StoreManagerFactory().create(manager='mongodb')
    await storage_manager.store_summary_feedback(form=form)
    return {'user': form.user, 'document_id': form.document_id}


@router.post("/summarize/stream")
async def stream_summarize(file: UploadFile = File(...)):
    return await trigger_sumamrization_service(file, execution_strategy='stream')


@router.post("/summarize/")
async def invoke_summarize(file: UploadFile = File(...)):
    return await trigger_sumamrization_service(file, execution_strategy='invoke')


async def trigger_sumamrization_service(file: UploadFile, execution_strategy: str):
    contents = await file.read()

    with NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(contents)
        tmp_file.flush()
        tmp_file.seek(0)  # Ensure the file pointer is at the start

        service = (
            SUMARIZERS['simple']()
            .set_loader(file_type=magic.from_buffer(contents, mime=True), file_path=tmp_file.name)
            .set_chatmodel(service='ollama', model='llama3.1')
            .set_execution_strategy(execution_strategy)
            .build()
        )

        return await service.process_summary_generation()
