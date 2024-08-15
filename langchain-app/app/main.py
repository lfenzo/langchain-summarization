from fastapi import FastAPI

from app.routers.summarize import router as summarization_router

app = FastAPI()

app.include_router(summarization_router)
