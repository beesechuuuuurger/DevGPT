from langchain.document_loaders import (
    TextLoader, UnstructuredHTMLLoader, UnstructuredPowerPointLoader, DataFrameLoader, PyPDFLoader, 
    NotebookLoader, CSVLoader, UnstructuredFileLoader, UnstructuredURLLoader, UnstructuredImageLoader, DirectoryLoader
)
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

import os
from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env.

openai_api_key = os.getenv('OPENAI_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
google_cse_id = os.getenv('GOOGLE_CSE_ID')


def load_document(self, document_path, source_type=None):
    # Get the file extension
    _, file_extension = os.path.splitext(document_path)

    # If source_type is not provided, infer it from the file extension
    if source_type is None:
        source_type = file_extension.lstrip('.')  # remove the leading dot

    # Select the appropriate loader based on the source type
    if source_type == 'txt':
        loader = TextLoader()
    elif source_type == 'html':
        loader = UnstructuredHTMLLoader()
    elif source_type == 'pptx':
        loader = UnstructuredPowerPointLoader()
    elif source_type == 'pdf':
        loader = PyPDFLoader()
    elif source_type == 'ipynb':
        loader = NotebookLoader()
    elif source_type == 'csv':
        loader = CSVLoader()
    elif source_type == 'unstructured_file':
        loader = UnstructuredFileLoader()
    elif source_type == 'url':
        loader = UnstructuredURLLoader()
    elif source_type == 'image':
        loader = UnstructuredImageLoader()
    elif source_type == 'directory' or 'folder':
        loader = DirectoryLoader()
    else:
        raise ValueError(f"Unsupported source type: {source_type}")

    # Load the document using the selected loader
    document = loader.load(document_path)
    return document

def split_document(self, document):
    # Split the document into chunks
    chunks = self.text_splitter.split_text(document)
    return chunks

def generate_embeddings(self, chunks):
    # Generate embeddings for each chunk
    embeddings = [self.embeddings.embed_query(chunk) for chunk in chunks]
    return embeddings

def process_document(self, document_path):
    # Load the document
    document = self.load_document(document_path)

    # Split the document into chunks
    chunks = self.split_document(document)

    # Generate embeddings for each chunk
    embeddings = self.generate_embeddings(chunks)

    return embeddings
