from typing import Optional

from langchain.pydantic_v1 import BaseModel, Field


class DocumentInfo(BaseModel):
    """General document information to better guide the summary generation process."""
    text_type: Optional[str] = Field(
        """
        The type of the text, examples: "article", "report", "blog post", "academic paper", etc
        """
    )
    media_type: Optional[str] = Field(
        """
        The media type, examples: "extracted pdf file", "audio transcript", "slide presentation",
        "text file", etc
        """
    )
    document_domain: Optional[str] = Field(
        """
        The document domain, examples: "technical", "medical", "legal", "financial", etc
        """
    )
    audience: Optional[str] = Field(
        """
        The document target audience, examples: "engineers", "students", "researchers", "academics"
        , etc
        """
    )
    audience_expertise: Optional[str] = Field(
        """
        The audience expertise level based on the text content, examples: "beginner",
        "intermediete", "expert", etc
        """
    )
    key_points: Optional[str] = Field(
        """
        Key sections or aspects the audience is likely interested in (e.g., methods in a scientific
        paper, key results in a financial report)
        """
    )
