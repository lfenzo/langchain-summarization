from typing import Iterator, Optional

from langchain.chat_models.base import BaseChatModel
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.summarizers.ollama.ollama_summarizer import OllamaSummarizer


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


class DynamicPromptSummarizer(OllamaSummarizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def extraction_model(self) -> BaseChatModel:
        return ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

    @property
    def extraction_prompt(self):
        return ChatPromptTemplate.from_messages([
            (
                "human",
                """
                You are an expert extraction algorithm, specialized in extracting structured
                information. Your task is to accurately identify and extract relevant attributes
                from the provided text. For each attribute, if the value cannot be determined from
                the text, return 'null' as the attribute's value. Ensure the extracted information
                is concise, relevant, and structured according to the required format."
                """
            ),
            ("human", "{text}"),
        ])

    @property
    def summarization_prompt(self):
        return ChatPromptTemplate.from_messages([
            ('human', "You an expert AI multi-lingual summary writer."),
            ('human', "Just write the summary, no need for introduction phrases."),
            ('human', "The summary must contain ~25% of the length of the original text."),
            ('human', "Ensure that the summary is in the same language as the original."),
            (
                'human',
                """
                Here are further information to guide you when generating the summary. Make sure
                that all these points are taken in consideretion in the appropriate summary:
                - Text Type: {text_type}
                - Media Type: {media_type}
                - Document Domain: {document_domain}
                - Audience: {audience}
                - Audience Expertise: {audience}
                - Document Key Points: {key_points}
                """
            ),
            ('human', "{text}"),
        ])

    def render_summary(self, content) -> Iterator:
        text = self._get_text_from_content(content=content)

        combined_chain = (
            self.extraction_prompt
            | self.extraction_model.with_structured_output(schema=DocumentInfo)
            | (lambda doc_info: {
                "text": text,
                "text_type": doc_info.text_type,
                "media_type": doc_info.media_type,
                "document_domain": doc_info.document_domain,
                "audience": doc_info.audience,
                "audience_expertise": doc_info.audience_expertise,
                "key_points": doc_info.key_points,
            })
            | self.summarization_prompt
            | self.model
        )

        return combined_chain.astream({"text": text})
