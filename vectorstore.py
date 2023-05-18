from getpass import getpass
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import (
    TextLoader, UnstructuredHTMLLoader, UnstructuredPowerPointLoader, DataFrameLoader, PyPDFLoader, 
    NotebookLoader, CSVLoader, UnstructuredFileLoader, UnstructuredURLLoader, UnstructuredImageLoader, DirectoryLoader
)

class LocalIndex:
    def __init__(self, api_key, persist_directory='D:\\DEV\\codeGPT\\codegpt\\vectorstore', chat_history_file='D:\\DEV\\codeGPT\\codegpt\\vectorstore\\chat_history.txt'):
        self.api_key = api_key
        os.environ["OPENAI_API_KEY"] = self.api_key
        self.persist_directory = persist_directory
        self.chat_history_file = chat_history_file
        self.embedding = OpenAIEmbeddings()
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.db = None

    def load_chat_history(self):
        with open(self.chat_history_file, 'r') as f:
            return f.read().splitlines()

    def add_to_chat_history(self, message):
        with open(self.chat_history_file, 'a') as f:
            f.write(message + '\n')

    def load_documents_from_source(self, source, source_type):
        if source_type == 'text':
            loader = TextLoader(source)
        elif source_type == 'html':
            loader = UnstructuredHTMLLoader(source)
        elif source_type == 'powerpoint':
            loader = UnstructuredPowerPointLoader(source)
        elif source_type == 'dataframe':
            loader = DataFrameLoader(source)
        elif source_type == 'pdf':
            loader = PyPDFLoader(source)
        elif source_type == 'jupyter_notebook':
            loader = NotebookLoader(source)
        elif source_type == 'csv':
            loader = CSVLoader(source)
        elif source_type == 'unstructured_file':
            loader = UnstructuredFileLoader(source)
        elif source_type == 'url':
            loader = UnstructuredURLLoader(source)
        elif source_type == 'image':
            loader = UnstructuredImageLoader(source)
        elif source_type == 'directory':
            loader = DirectoryLoader(source)
        else:
            raise ValueError(f'Unsupported source type: {source_type}')

        documents = loader.load()
        docs = self.text_splitter.split_documents(documents)
        return docs

    def create_index(self, docs):
        self.db = Chroma.from_documents(docs, self.embedding, persist_directory=self.persist_directory)
        self.db.persist()

    def load_index(self):
        self.db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding)

    def similarity_search(self, query):
        return self.db.similarity_search(query)

    def similarity_search_with_score(self, query):
        return self.db.similarity_search_with_score(query)
