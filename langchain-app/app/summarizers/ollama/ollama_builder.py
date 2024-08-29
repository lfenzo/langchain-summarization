from app.summarizers.base.base_builder import SummarizerBuilder
from app.summarizers.ollama.ollama_summarizer import OllamaSummarizer


class OllamaSummarizationBuilder(SummarizerBuilder):
    def __init__(self):
        super().__init__()
        self.base_url = "http://ollama-server:11434"
        self.model_name = "gemma2:27b"

    def build(self):
        return OllamaSummarizer(**self.get_init_params())

    def get_init_params(self):
        params = {
            "base_url": self.base_url,
            "model_name": self.model_name,
        }
        params.update(super().get_init_params())
        return params

    def set_base_url(self, base_url: str):
        self.base_url = base_url
        return self

    def set_model(self, model_name: str):
        self.model_name = model_name
        return self
