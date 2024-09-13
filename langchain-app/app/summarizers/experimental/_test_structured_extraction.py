"""
Set of test function prototyping various methods to improve summary generation.
"""

from time import sleep
from typing import Optional

from langchain.chat_models.base import BaseChatModel
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI


OLLAMA_SERVER = 'http://ollama-server:11434'


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


class InfoPydantic(BaseModel):
    """Text, content and reader characteristics."""

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


def _accumulate_text(document):
    text = ""
    for page in document:
        text += page.page_content + "\n"
    return text


def base_summarization_prompt(text: str, model: BaseChatModel):
    """
    Tests with the base summarization prompt (no tools, structured extraction, etc)
    """
    prompt = ChatPromptTemplate.from_messages([
        ('human', "You are an expert in writing concise and accurate summaries in multiple languages."),
        ('human', "Your task is to generate a summary of the provided text."),
        ('human', "Start the summary immediately; do not include any introductory phrases."),
        ('human', "The summary should be approximately 25% of the length of the original text."),
        ('human', "Ensure the summary is written in the same language as the original text."),
        ('human', "Write in a direct style, avoiding phrases like 'the text states'"),
        ("human", "focus solely on conveying the key facts."),
        ('human', "Consider the intended audience of the document when generating the summary."),
        ('human', "{text}"),
    ])

    chain = prompt | model

    summary = ""
    for chunk in chain.stream({'text': text}):
        print(chunk.content, end='', flush=True)
        summary += chunk.content

    return summary


def structured_extraction_1(text: str, model: BaseChatModel, extraction_model: BaseChatModel):
    """
    1. Get the reader type (what is the target audience for the original document)
    2. Generate 3 relevant bullet points that a reader of the detected type considers relevant
    3. Generate the summary using the reader type and relevant points
    """
    audience_extraction_prompt = ChatPromptTemplate.from_messages([
        (
            "human",
            """
            You are an expert extraction algorithm, specialized in identifying and extracting the
            target audience of technical and non-technical documents. Your task is to accurately
            identify and extract a short, concise phrase that describes the most probable readers
            or users of the provided text. Ensure the extracted information is clear, relevant, and
            concise.
            """
        ),
        ("human", "{text}"),
    ])

    audience_main_points_prompt = ChatPromptTemplate.from_messages([
        (
            "human",
            """
            You will be given document target audience, generate exactly 3 bullet points that
            representing concisely the main characteristics a reader from that audience considers
            relevant in a summary.
            No need for explanation or introduction, just output the list with 3 items.
            Example:
            - techincal specifications
            - information on how to use the device
            - how to access the informatino collected
            """
        ),
        ("human", "{audience}"),
    ])

    summary_prompt = ChatPromptTemplate.from_messages([
        (
            "human",
            """
            You are an expert in writing concise and accurate summaries in several languages.
            Given the text, write a summary considering that the target audience for that document
            is {audience}.
            Do know that these readers consider relevant {main_points}, so make sure they are
            well covered in your summary.
            No need for introduction or justification, just go straight to the summary.
            Make sure that the summary is in the same language as the original.
            """
        ),
        ("human", "{text}"),
    ])

    audience_chain = (
        audience_extraction_prompt | extraction_model.with_structured_output(schema=InfoPydantic)
    )
    audience = audience_chain.invoke({"text": text}).audience
    print(audience)
    print("=" * 100)

    points_chain = audience_main_points_prompt | model
    main_points = points_chain.invoke({"audience": audience})
    print(main_points)
    print("=" * 100)

    print("aguardando o tempo para chama novamente o modelo")
    sleep(60)
    summary_chain = summary_prompt | model | StrOutputParser()

    for chunk in summary_chain.stream({
        "audience": audience,
        "main_points": main_points,
        "text": text,
    }):
        print(chunk, end='', flush=True)


def structured_extraction_2(text: str, model: BaseChatModel, extraction_model: BaseChatModel):
    """
    1. Get in one pass the audience, document type and main topics via structured extraction
    2. Insert the collected information into a teplate
    3. Generate the summary using that template
    """
    extraction_prompt = ChatPromptTemplate.from_messages([
        (
            "human",
            """
            You are an expert extraction algorithm, specialized in extracting structured
            information. Your task is to accurately identify and extract relevant attributes from
            the provided text. For each attribute, if the value cannot be determined from the text,
            return 'null' as the attribute's value. Ensure the extracted information is concise,
            relevant, and structured according to the required format."
            Example:
            [
                "audience": "researchers in renewable energy",
                "main_topic": "Applied Machine Learning in solar energy solutions",
                "document_type": "research paper",
            ]
            """
        ),
        ("human", "{text}"),
    ])

    summarization_prompt = ChatPromptTemplate.from_messages([
        ('human', "You an expert AI multi-lingual summary writer."),
        ('human', "Just write the summary, no need for introduction phrases."),
        (
            'human',
            "Consider that the summary is targeted towards {audience} reader(s). "
            "Consider that the main topic in this document is {main_topic} and that this "
            "is a {document_type} document. Generate a summary appropriate to this "
            "audience, document format and main topic."
        ),
        ('human', "The summary must contain ~25% of the length of the original text."),
        ('human', "Ensure that the summary is in the same language as the original."),
        ('human', "{text}"),
    ])

    extracton_chain = (
        extraction_prompt
        | extraction_model.with_structured_output(schema=InfoPydantic)
    )
    summarization_chain = summarization_prompt | model | StrOutputParser()

    extracted_info = extracton_chain.invoke({"text": text})
    print(extracted_info)
    print("=" * 100)

    for chunk in summarization_chain.stream({
        "text": text,
        "audience": extracted_info.audience,
        "main_topic": extracted_info.main_topic,
        "document_type": extracted_info.document_type
    }):
        print(chunk, end='', flush=True)


def structured_extraction_3(text: str, model: BaseChatModel, extraction_model: BaseChatModel):
    """
    1. Get in one pass more detailed data about the summary (see DocumentInfo class)
    2. Insert the collected information into a teplate acoomodating the pydantic class
    3. Generate the summary using that template
    """
    extraction_prompt = ChatPromptTemplate.from_messages([
        (
            "human",
            """
            You are an expert extraction algorithm, specialized in extracting structured
            information. Your task is to accurately identify and extract relevant attributes from
            the provided text. For each attribute, if the value cannot be determined from the text,
            return 'null' as the attribute's value. Ensure the extracted information is concise,
            relevant, and structured according to the required format."
            """
        ),
        ("human", "{text}"),
    ])

    summarization_prompt = ChatPromptTemplate.from_messages([
        ('human', "You an expert AI multi-lingual summary writer."),
        ('human', "Just write the summary, no need for introduction phrases."),
        ('human', "The summary must contain ~25% of the length of the original text."),
        ('human', "Ensure that the summary is in the same language as the original."),
        (
            'human',
            """
            Here are further information to guide you when generating the summary. Make sure that
            all these points are taken in consideretion in the appropriate summary:
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

    extracton_chain = (
        extraction_prompt
        | extraction_model.with_structured_output(schema=DocumentInfo)
    )
    summarization_chain = summarization_prompt | model | StrOutputParser()

    extracted_info = extracton_chain.invoke({"text": text})
    print(extracted_info)
    print("=" * 100)
    sleep(60)

    for chunk in summarization_chain.stream({
        "text": text,
        "text_type": extracted_info.text_type,
        "media_type": extracted_info.media_type,
        "document_domain": extracted_info.document_domain,
        "audience": extracted_info.audience,
        "audience_expertise": extracted_info.audience_expertise,
        "key_points": extracted_info.key_points,
    }):
        print(chunk, end='', flush=True)


def structured_extraction_inversed(
    text: str,
    model: BaseChatModel,
    extraction_model: BaseChatModel,
):
    summary = base_summarization_prompt(text=text, model=model)
    extraction_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an expert extraction algorithm, specialized in extracting structured
            information. Your task is to accurately identify and extract relevant attributes from
            the provided text. For each attribute, if the value cannot be determined from the text,
            return 'null' as the attribute's value. Ensure the extracted information is concise,
            relevant, and structured according to the required format
            Example:
            [
                "audience": "researchers in renewable energy",
                "main_topic": "Applied Machine Learning in solar energy solutions",
                "document_type": "research paper",
            ]
            """
        ),
        ("human", "{text}"),
    ])

    summarization_prompt = ChatPromptTemplate.from_messages([
        ('human', "You an expert AI multi-lingual summary writer."),
        ('human', "Just write the summary, no need for introduction phrases."),
        (
            'human',
            "Consider that the summary is targeted towards {audience} reader(s). "
            "Consider that the main topic in this document is {main_topic} and that this "
            "is a {document_type} document. Generate a summary appropriate to this "
            "audience, document format and main topic."
        ),
        ('human', "The summary must contain ~25% of the length of the original text."),
        ('human', "Ensure that the summary is in the same language as the original."),
        ('human', "{text}"),
    ])

    extracton_chain = (
        extraction_prompt
        | extraction_model.with_structured_output(schema=InfoPydantic)
    )
    summarization_chain = summarization_prompt | model | StrOutputParser()

    extracted_info = extracton_chain.invoke({"text": summary})
    print(extracted_info)
    print("=" * 100)

    print("aguardando o tempo para chama novamente o modelo")
    sleep(60)
    for chunk in summarization_chain.stream({
        "text": text,
        "audience": extracted_info.audience,
        "main_topic": extracted_info.main_topic,
        "document_type": extracted_info.document_type
    }):
        print(chunk, end='', flush=True)


def manual_extraction_1(text: str, structured_type: str = 'pydantic', model: str = "llama3.1"):
    """
    1. Extract the audience manually (prompt + model call)
    2. Extract the main points manually (prompt + model call)
    3. Extract the document type manually (prompt + model call)
    4. Generate the summary based on the information extracted manually
    """
    audience_prompt = ChatPromptTemplate.from_messages([
        ('system', "Receive the text and output ONLY a single, small phrase containing the target audience of that text"),
        ('system', "No need for introduction, just generate the requested phrase"),
        ('system', "Our output must contain at most 8 words"),
        ('human', "\n\n{text}"),
    ])
    model = ChatOllama(model=model, base_url=OLLAMA_SERVER)

    chain = audience_prompt | model

    for chunk in chain.stream({"text": text}):
        print(chunk.content, end='', flush=True)


def load_text(path: str) -> str:
    with open(path, 'r') as file:
        return file.read()


if __name__ == "__main__":
    file_path = 'input/tdd.pdf'
    loader = PyMuPDFLoader(file_path=file_path)
    text = _accumulate_text(loader.load())
    # text = load_text(path=file_path)

    # model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    model = ChatOllama(model='gemma2:27b', base_url=OLLAMA_SERVER)
    # extraction_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
    extraction_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

    # base_summarization_prompt(text=text, model=model)
    # structured_extraction_1(text=text, model=model, extraction_model=extraction_model)
    # structured_extraction_2(text=text, model=model, extraction_model=extraction_model)
    structured_extraction_3(text=text, model=model, extraction_model=extraction_model)
    # structured_extraction_inversed(text, model=model, extraction_model=extraction_model)
