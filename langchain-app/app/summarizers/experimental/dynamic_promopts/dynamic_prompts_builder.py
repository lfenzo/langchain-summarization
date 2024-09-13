from app.summarizers.ollama.ollama_builder import OllamaSummarizationBuilder
from app.summarizers.experimental.dynamic_promopts.dynamic_prompts_summarizer import (
    DynamicPromptSummarizer
)


class DynamicPromptSummarizerBuilder(OllamaSummarizationBuilder):
    def build(self):
        return DynamicPromptSummarizer(**self.get_init_params())
