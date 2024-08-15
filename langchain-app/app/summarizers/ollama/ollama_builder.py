from app.summarizers.base.base_builder import SummarizerBuilder
from app.summarizers.ollama.ollama_summarizer import OllamaSummarizer


class OllamaSummarizationBuilder(SummarizerBuilder):
    def __init__(self):
        super().__init__()
        self.base_url = "http://ollama-server:11434"
        self.model = "gemma2:27b"

    def set_base_url(self, base_url: str):
        self.base_url = base_url
        return self

    def set_model(self, model: str):
        self.model = model
        return self

    def build(self):
        base_params = super().build()
        return OllamaSummarizer(model=self.model, base_url=self.base_url, **base_params)
