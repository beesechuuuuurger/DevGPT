from langchain.document_loaders import (
    TextLoader, UnstructuredHTMLLoader, UnstructuredPowerPointLoader, DataFrameLoader, PyPDFLoader, 
    NotebookLoader, CSVLoader, UnstructuredFileLoader, UnstructuredURLLoader, UnstructuredImageLoader, DirectoryLoader
)
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

import os

class DocumentEmbeddingProcessor:
    def __init__(self):
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.embeddings = OpenAIEmbeddings()

    def load_document(self, document_path):
        # Get the file extension
        _, file_extension = os.path.splitext(document_path)

        # Select the appropriate loader based on the file extension
        if file_extension == '.txt':
            loader = TextLoader()
        elif file_extension == '.html':
            loader = UnstructuredHTMLLoader()
        elif file_extension == '.pptx':
            loader = UnstructuredPowerPointLoader()
        elif file_extension == '.pdf':
            loader = PyPDFLoader()
        elif file_extension == '.ipynb':
            loader = NotebookLoader()
        elif file_extension == '.csv':
            loader = CSVLoader()
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

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
