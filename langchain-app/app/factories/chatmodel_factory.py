from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_ollama import ChatOllama


class ChatModelFactory:

    def __init__(self) -> None:
        self.available_chatmodels = {
            'google-genai': ChatGoogleGenerativeAI,
            'google-vertex': ChatVertexAI,
            'ollama': ChatOllama,
        }

    def create(self, chatmodel: str, **kwargs) -> BaseChatModel:
        return self.available_chatmodels[chatmodel](**kwargs)
