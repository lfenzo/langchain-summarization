from app.summarizers.ollama.ollama_builder import OllamaSummarizationBuilder
from app.summarizers.experimental.document_centered.document_centered_summarizer import (
    DocumentCenteredSummarizer
)


class DocumentCenteredSummarizationBuilder(OllamaSummarizationBuilder):
    def build(self):
        return DocumentCenteredSummarizer(**self.get_init_params())
