from pydantic import BaseModel


class SummarizeResponse(BaseModel):
    file_name: str
    summary: str
