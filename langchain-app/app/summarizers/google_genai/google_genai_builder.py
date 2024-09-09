from app.summarizers.base.base_builder import SummarizerBuilder
from app.summarizers.google_genai.google_genai_summarizer import GoogleGenAISummarizer


class GoogleGenAISummarizerBuilder(SummarizerBuilder):
    def __init__(self):
        super().__init__()
        self.model_name = "gemini-1.0-pro-latest"
        self.location = "us-central1"

    def build(self):
        return GoogleGenAISummarizer(**self.get_init_params())

    def get_init_params(self):
        params = {
            "model_name": self.model_name,
            "location": self.location,
        }
        params.update(super().get_init_params())
        return params

    def set_location(self, location: str):
        self.location = location
        return self

    def set_model_name(self, model_name: str):
        self.model_name = model_name
        return self
