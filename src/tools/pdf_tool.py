from langchain_community.document_loaders import PyPDFLoader
from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Literal, List
from langchain_core.documents import Document  

class PdfInput(BaseModel):
    file_path: str = Field(description="The path to the PDF file to be loaded.")
    password: str | bytes | None = Field(default=None, description="Optional password for opening encrypted PDFs.")
    headers: dict | None = Field(default=None, description="Optional headers to use for GET request to download a file from a web path.")
    extract_images: bool = Field(default=False, description="Whether to extract images from the PDF.")
    mode: Literal['single', 'page'] = Field(default='page', description='The extraction mode, either "single" for the entire document or "page" for page-wise extraction.')
    images_inner_format: Literal['text', 'markdown-img', 'html-img'] = Field(
        default='text',
        description='''The format for the parsed output.
        "text" = return the content as is
        "markdown-img" = wrap the content into an image markdown link, w/ link pointing to (![body)(#)]
        "html-img" = wrap the content as the alt text of an tag and link to (<img alt="{body}" src="#"/>)'''
    )
    extraction_mode: Literal['plain', 'layout'] = "plain"
    extraction_kwargs: dict | None = None

def load_pdf(params: PdfInput) -> List[Document]:
    "Loads pdf"
    loader = PyPDFLoader(**params.model_dump())
    docs = loader.load()    
    return  docs


@tool('pdf_loader', 
      description="Use `pdf_loader` when you need to extract a pdf file.", 
      args_schema=PdfInput
      )     
def load_pdf_tool(params: PdfInput):
    return load_pdf(params)

