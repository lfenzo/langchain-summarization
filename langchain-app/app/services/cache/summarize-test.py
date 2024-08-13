from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import WebBaseLoader
from langchain_ollama import ChatOllama

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

llm = ChatOllama(model="gemma2:27b", base_url="http://ollama-server:11434")
chain = load_summarize_chain(llm, chain_type="stuff")

print(chain.run(docs))
