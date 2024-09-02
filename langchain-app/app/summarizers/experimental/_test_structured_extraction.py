"""
Set of test function prototyping various methods to improve summary generation.
"""

from typing import TypedDict, Annotated, Optional

from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredMarkdownLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama


OLLAMA_SERVER = 'http://ollama-server:11434'


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


class InfoTypedDict(TypedDict):
    """Text, content and reader characteristics."""

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


def _accumulate_text(document):
    text = ""
    for page in document:
        text += page.page_content + "\n"
    return text


def base_summarization_prompt(text: str, model: str = "llama3.1"):
    """
    Tests with the base summarization prompt (no tools, structured extraction, etc)
    """
    prompt = ChatPromptTemplate.from_messages([
        ('system', "You are an expert in writing concise and accurate summaries in multiple languages."),
        ('system', "Your task is to generate a summary of the provided text."),
        ('system', "Start the summary immediately; do not include any introductory phrases."),
        ('system', "The summary should be approximately 25% of the length of the original text."),
        ('system', "Ensure the summary is written in the same language as the original text."),
        ('system', "Write in a direct style, avoiding phrases like 'the text states'; focus solely on conveying the key facts."),
        ('system', "Consider the intended audience of the original document when crafting your summary."),
        ('human', "{text}"),
    ])

    model = ChatOllama(model=model, base_url=OLLAMA_SERVER)
    chain = prompt | model

    for chunk in chain.stream({'text': text}):
        print(chunk.content, end='', flush=True)


def structured_extraction_1(structured_type: str = 'pydantic', model: str = "llama3.1"):
    """
    1. Get the reader type (what is the target audience for the original document)
    2. Generate 3 relevant bullet points that a reader of the detected type considers relevant
    3. Generate the summary using the reader type and relevant points
    """
    audience_extraction_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert extraction algorithm, specialized in identifying and extracting the target "
            "audience of technical and non-technical documents. Your task is to accurately identify and "
            "extract a short, concise phrase that describes the most probable readers or users of the "
            "provided text. Ensure the extracted information is clear, relevant, and concise.",
        ),
        ("human", "{text}"),
    ])
    extraction_model = ChatOllama(model='llama3-groq-tool-use', base_url=OLLAMA_SERVER, temperature=0)
    # summarization_model = ChatOllama(model=model, base_url=OLLAMA_SERVER)

    extracton_chain = (
        audience_extraction_prompt
        | extraction_model.with_structured_output(schema=InfoPydantic, include_raw=True)
    )

    extracted_info = extracton_chain.invoke({"text": text})
    print(extracted_info)


def structured_extraction_2(text: str, structured_type: str = 'pydantic', model: str = "llama3.1"):
    """
    1. Get in one pass the audience, document type and main topics via structured extraction
    2. Insert the collected information into a teplate
    3. Generate the summary using that template
    """
    extraction_model = ChatOllama(model='llama3.1', base_url=OLLAMA_SERVER, temperature=0)
    summarization_model = ChatOllama(model=model, base_url=OLLAMA_SERVER)

    extraction_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert extraction algorithm, specialized in extracting structured information. "
            "Your task is to accurately identify and extract relevant attributes from the provided text. "
            "For each attribute, if the value cannot be determined from the text, return 'null' as the attribute's value. "
            "Ensure the extracted information is concise, relevant, and structured according to the required format.",
        ),
        ("human", "{text}"),
    ])

    summarization_prompt = ChatPromptTemplate.from_messages([
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
        ('human', "{text}"),
    ])

    extracton_chain = extraction_prompt | extraction_model.with_structured_output(schema=InfoTypedDict, include_raw=True)
    summarization_chain = summarization_prompt | summarization_model | StrOutputParser()

    extracted_info = extracton_chain.invoke({"text": text})
    print(extracted_info)

    for chunk in summarization_chain.stream({
        "text": text,
        "audience": extracted_info.audience,
        "main_topic": extracted_info.main_topic,
        "document_type": extracted_info.document_type
    }):
        print(chunk.content, end='', flush=True)


def manual_extraction_1(text: str, structured_type: str = 'pydantic', model: str = "llama3.1"):
    """
    1. Extract the audience manually (prompt + model call)
    2. Extract the main points manually (prompt + model call)
    3. Extract the document type manually (prompt + model call)
    4. Generate the summary based on the information extracted manually
    """
    audience_prompt = ChatPromptTemplate.from_messages([
        ('system', "Receive the text and ouput ONLY a single, small phrase containing the target audience of that text"),
        ('system', "No need for introduction, just generate the requested phrase"),
        ('system', "Our output must contain at most 5 words"),
        ('human', "\n\n{text}"),
    ])
    model = ChatOllama(model=model, base_url=OLLAMA_SERVER)

    chain = audience_prompt | model

    for chunk in chain.stream({"text": text}):
        print(chunk.content, end='', flush=True)


if __name__ == "__main__":
    file_path = 'input/digital-thermometer-ds18b20.pdf'
    loader = PyMuPDFLoader(file_path=file_path)
    text = _accumulate_text(loader.load())
    print(text)

    #base_summarization_prompt(text)
    structured_extraction_1(text, model="llama3-groq-tool-use")
