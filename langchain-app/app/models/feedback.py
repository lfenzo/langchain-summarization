from typing import Optional

from pydantic import BaseModel


class FeedbackForm(BaseModel):
    user: str
    document_id: str
    feedback: Optional[str] = None
    written_feedback: Optional[str] = None
