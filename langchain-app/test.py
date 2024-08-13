#from langchain.chains.summarize import load_summarize_chain
#from langchain_community.document_loaders import WebBaseLoader
#from langchain_ollama import ChatOllama
#
#from typing_extensions import Annotated, TypedDict
#
#
## TypedDict
#class Summary(TypedDict):
#    """Joke to tell user."""
#
#    description: Annotated[str, ..., "Quick description of the text."]
#    summary: Annotated[int, ..., "The summary of the original text."]
#
#
#loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
#docs = loader.load()
#
#llm = ChatOllama(model='llama3.1', base_url="http://ollama-server:11434")
#structured = llm.with_structured_output(Summary)
#print(docs)
#chain = load_summarize_chain(llm, chain_type="stuff")
#
#async for event in chain.astream_events():
#
#for a in chain.stream(docs):
#    print(a)
#    print("\n\n")
from typing import List
from pydantic import BaseModel, Field
from langchain.chains.summarize import load_summarize_chain
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import WebBaseLoader


class SummaryResult(BaseModel):
    summary: str = Field(..., description="The summary of the input text")
    key_points: List[str] = Field(..., description="The key points extracted from the input text")


loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()
llm = ChatOllama(model='llama3.1', base_url="http://ollama-server:11434")
#chain = load_summarize_chain(llm, chain_type="stuff")
chain = load_summarize_chain(llm, chain_type="map_reduce")

print(chain.invoke({"input_documents": docs})['output_text'])

for a in chain.stream({"input_documents": docs}):
    print(a['output_text'])
    print("\n\n")
