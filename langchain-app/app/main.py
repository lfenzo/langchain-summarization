from fastapi import FastAPI

from app.routers.summarize import router as summarization_router

app = FastAPI()

app.include_router(summarization_router)

#from fastapi import FastAPI, Body, Request, UploadFile, File
#from langserve import add_routes
#from pydantic import BaseModel
#from langchain.prompts import ChatPromptTemplate
#from langchain_ollama import ChatOllama
#from typing import Optional
#
#from runnables import joke
#
#
#app = FastAPI()
#
#
#add_routes(
#    app,
#    runnable=joke.get_runnable(),
#    path="/joke"
#)
#
#
#class SummarizeRequest(BaseModel):
#    file_name: str
#    file_bytes: Optional[bytes]
#
#
#@app.post("/summarize")
#async def summarize(file: UploadFile = File(...)):
#    contents = await file.read()
#    print(f"Received file: {file.filename}, size: {len(contents)} bytes")
#    return {"filename": file.filename, "size": len(contents)}
#
#
#@app.post("/summarize1")
#async def summarize1():
#    model = ChatOllama(model="gemma2:27b", base_url="http://ollama-server:11434")
#    prompt = ChatPromptTemplate.from_messages([
#        ('system', "You are a multi-language expert comedian"),
#        ('user', 'Tell me a joke about {topic} in {language}')
#    ])
#    return prompt | model
#
#    data = request.json
#    s3_url = data.get("url")
#
#    # Process the document using your chain
#    summary = my_chain.run({"document_url": s3_url})
#    return {"summary": summary}
