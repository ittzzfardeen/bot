from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv,dotenv_values
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import CSVLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



load_dotenv()

def pdf_file_load(data):
    loader=DirectoryLoader(
        data,
        glob="*.csv",
        loader_cls=CSVLoader
    )
    document=loader.load()
    return document

from typing import List
from langchain_core.documents import Document

def filter_to_minimal(docs:  List[Document]) -> List[Document]:
    minimal_docs :List[Document]= []
    for doc in docs:
        src=doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source":src}
            )
        )
    return minimal_docs


def text_split(minimal_docs):
    text_split=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=500)
    splits=text_split.split_documents(minimal_docs)
    return text_split

def downloading_huggingface():
    embedding_model=HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    return embedding_model
