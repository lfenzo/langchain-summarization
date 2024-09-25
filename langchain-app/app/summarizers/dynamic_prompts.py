from typing import Any, AsyncIterator, Dict

from langchain_core.documents.base import Document
from langchain_core.messages.ai import AIMessageChunk, AIMessage
from langchain.chat_models.base import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from app.models import DocumentInfo
from app.summarizers import BaseSummarizer


class DynamicPromptSummarizer(BaseSummarizer):

    def __init__(
        self,
        chatmodel: BaseChatModel,
        extraction_chatmodel: BaseChatModel,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.chatmodel = chatmodel
        self.extraction_chatmodel = extraction_chatmodel

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

    @property
    def extraction_chain(self):
        return (
            self.extraction_prompt
            | self.extraction_chatmodel.with_structured_output(schema=DocumentInfo)
        )

    @property
    def summarization_chain(self):
        return self.summarization_prompt | self.chatmodel

    def summarize(self, content: list[Document]) -> AsyncIterator[AIMessageChunk] | AIMessage:
        text = self._get_text_from_content(content=content)
        structured_information = self.extraction_chain.invoke({"text": text})
        return self.execution_strategy.run(
            runnable=self.summarization_chain,
            kwargs={"text": text, **structured_information.dict()},
        )

    def get_metadata(self, file: str, generation_metadata: Dict) -> Dict[str, Any]:
        metadata = self._get_base_metadata(file=file, generation_metadata=generation_metadata)
        metadata.update({
            'chatmodel': repr(self.chatmodel),
            "summarization_prompt": repr(self.summarization_prompt),
            'extraction_chatmodel': repr(self.extraction_chatmodel),
            "extraction_prompt": repr(self.extraction_prompt),
            "structured_straction_schema": DocumentInfo.__class__.__name__,
        })
        return metadata
