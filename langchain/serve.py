from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langserve import add_routes

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

llm = ChatOllama(
    model="llama3.1",
    base_url="http://ollama-server:11434",
)

add_routes(
    app,
    llm,
    path="/openai",
)

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
add_routes(
    app,
    prompt | llm,
    path="/joke",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
