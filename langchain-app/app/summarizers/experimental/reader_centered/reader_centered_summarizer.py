from typing import Iterator

from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.runnables.base import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.summarizers.ollama.ollama_summarizer import OllamaSummarizer


class TextReader(BaseModel):
    reader_type: str = Field(
        description="""
            What is the reader type (persona) for the provided text. For example,
            "Engineer", "Physician", "teacher", "high school student", "college undergrad", etc
            (may include other types of readers as well).
            Obtain who the text is addressed to.
        """
    )


class ReaderCenteredSummarizer(OllamaSummarizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def extraction_model(self):
        return ChatOllama(model='mistral-nemo', base_url=self.base_url, cache=self.cache)

    @property
    def reader_type_prompt(self):
        return ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert extraction algorithm. "
                "Only extract relevant information from the text. "
                "If you do not know the value of an attribute asked to extract, "
                "return null for the attribute's value.",
            ),
            ("human", "{text}"),
        ])

    @property
    def reader_main_points_prompt(self):
        return ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are a helpful AI assistant.
                Based on an reader type, provide a list of excactly 3 bullet points consisting
                of the most relevant characteristics that this reader type finds helpful when
                reading a summary.
                No need for introduction, just output the bullet points!
                No explanations, just output a simple and small phrase per bullet point.
                Example:
                'engineer':
                    - technical datails and specifications;
                    - overall mechanisms, methods of processes;
                    - clear instructions on how to use and operate some device."
                """
            ),
            ("human", "{reader_type}"),
        ])

    @property
    def summarization_prompt(self):
        return ChatPromptTemplate.from_messages([
            ('system', "You produce high quality summaries in several languages"),
            ('system', "Produce the summary in the same language as the original text."),
            ('system', "Just write the summary, no need for introduction phrases."),
            (
                'system',
                "Consider that the summary is targeted towards {reader_type} reader(s). "
                "This type of reader finds the following relevant: {main_points} "
                "Make sure that these points are well highlighted in the summary."
            ),
            ('user', "Here is the text to be summarized:\n\n{text}"),
        ])

    def create_runnable(self) -> Runnable:
        return self.prompt | self.model | StrOutputParser()

    def render_summary(self, content) -> Iterator:
        reader_type_chain = (
            self.reader_type_prompt
            | self.extraction_model.with_structured_output(schema=TextReader)
        )

        reader_points_chain = self.reader_main_points_prompt | self.model

        summary_chain = self.summarization_prompt | self.model | StrOutputParser()

        text = ""
        for page in content:
            text += page.page_content + "\n"

        reader_type = reader_type_chain.invoke({"text": text})

        main_points = reader_points_chain.invoke({"reader_type": reader_type})

        return summary_chain.astream(
            {"text": text, "reader_type": reader_type, "main_points": main_points}
        )
