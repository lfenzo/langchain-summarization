from typing import Iterator, Optional, TypedDict, Annotated

from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.summarizers.ollama.ollama_summarizer import OllamaSummarizer


class DocumentInfo(BaseModel):
    document_type: Optional[str] = Field(
        description="""
            Type of the documet/text
            For example "scientific paper", "essay", "poem", "song lyrics", "recorded lecture"
            Feel free to include other types
        """
    )
    main_topic: Optional[str] = Field(
        description="""
            Main topic discussed along the document, presented in a small phase.
        """
    )
    audience: Optional[str] = Field(
        description="""
            Audience for the document, the most likely person to read the document
            For example: "engineer", "teacher", "student"
            Feel free to include other audiences
        """
    )


class Info(TypedDict):
    document_type: Annotated[
        str, None,
        """
        Type of the documet/text
        For example "scientific paper", "essay", "poem", "song lyrics", "recorded lecture"
        Feel free to include other types
        """
    ]
    main_topio = Annotated[
        str, None,
        """
        Main topic discussed along the document, presented in a small phase.
        """
    ]
    audiance = Annotated[
        str, None,
        """
        Audience for the document, the most likely person to read the document
        For example: "engineer", "teacher", "student"
        Feel free to include other audiences
        """
    ]


class DocumentCenteredSummarizer(OllamaSummarizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def extraction_model(self):
        return ChatOllama(
            model='llama3.1:70b', base_url=self.base_url, cache=self.cache, temperature=0,
        )

    @property
    def document_info_extraction_prompt(self):
        return ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are an expert extraction algorithm, specialized in extracting structured
                information. Your task is to accurately identify and extract relevant attributes
                from the provided text. For each attribute, if the value cannot be determined from
                the text, return 'null' as the attribute's value. Ensure the extracted information
                is concise, relevant, and structured according to the required format.
                """
            ),
            ("human", "{text}"),
        ])

    @property
    def summarization_prompt(self):
        return ChatPromptTemplate.from_messages([
            ('system', "You an expert AI multi-lingual summary writer."),
            ('system', "Just write the summary, no need for introduction phrases."),
            (
                'system',
                "Consider that the summary is targeted towards {audience} reader(s). "
                "Consider that the main topic in this document is {main_topic} and that this "
                "is a {document_type} document. Generate a summary appropriate to this "
                "audience, document format and main topic."
            ),
            ('system', "The summary must contain ~25% of the length of the original text."),
            ('system', "You must produce the summary in the same language as the original text."),
            ('human', "Text summarize:\n\n{text}"),
        ])

    def render_summary(self, content) -> Iterator:
        document_info_chain = (
            self.document_info_extraction_prompt
            | self.extraction_model.with_structured_output(schema=Info, include_raw=True)
        )

        summarization_chain = self.summarization_prompt | self.model

        text = ""
        for page in content:
            text += page.page_content + "\n"

        document_info = document_info_chain.invoke({"text": text})

        return summarization_chain.astream(
            {
                "text": text,
                "main_topic": document_info.main_topic,
                "audience": document_info.audience,
                "document_type": document_info.document_type
            }
        )
