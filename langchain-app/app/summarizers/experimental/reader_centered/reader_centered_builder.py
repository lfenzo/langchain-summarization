from app.summarizers.ollama.ollama_builder import OllamaSummarizationBuilder
from app.summarizers.experimental.reader_centered.reader_centered_summarizer import (
    ReaderCenteredSummarizer
)


class ReaderCenteredSummarizationBuilder(OllamaSummarizationBuilder):

    def build(self):
        return ReaderCenteredSummarizer(**self.get_init_params())
