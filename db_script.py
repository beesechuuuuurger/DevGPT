from getpass import getpass
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os

 # take environment variables from .env.

load_dotenv() 

class LocalIndex:
    def __init__(self, api_key=os.getenv('OPENAI_API_KEY'), persist_directory='D:\\DEV\\codeGPT\\codegpt\\vectorstore'):
        self.api_key = api_key
        os.environ["OPENAI_API_KEY"] = self.api_key
        self.persist_directory = persist_directory
        self.embedding = OpenAIEmbeddings()
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

        self.research_db = None
        self.documents_db = None


    def create_index(self, docs, index_type):
        if index_type == 'research_data':
            self.research_db = Chroma.from_documents(docs, self.embedding, persist_directory=self.persist_directory)
            self.research_db.persist()
        elif index_type == 'documents':
            self.documents_db = Chroma.from_documents(docs, self.embedding, persist_directory=self.persist_directory)
            self.documents_db.persist()
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

    def load_index(self, index_type):
        if index_type == 'research_data':
            self.research_db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding)
        elif index_type == 'documents':
            self.documents_db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

    def similarity_search(self, query, index_type):
        if index_type == 'research_data':
            return self.research_db.similarity_search(query)
        elif index_type == 'documents':
            return self.documents_db.similarity_search(query)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

    def similarity_search_with_score(self, query, index_type):
        if index_type == 'research_data':
            return self.research_db.similarity_search_with_score(query)
        elif index_type == 'documents':
            return self.documents_db.similarity_search_with_score(query)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
    def update_index(self, new_docs, index_type):
        # Load the new documents
        documents = [self.load_document(doc_path) for doc_path in new_docs]

        # Split the new documents into chunks
        chunks = [self.split_document(doc) for doc in documents]

        # Generate embeddings for the new chunks
        embeddings = [self.generate_embeddings(chunk) for chunk in chunks]

        # Load the existing index
        self.load_index(index_type)

        # Update the index with the new embeddings
        if index_type == 'research_data':
            self.research_db.add_documents(embeddings)
        elif index_type == 'documents':
            self.documents_db.add_documents(embeddings)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        # Persist the updated index
        if index_type == 'research_data':
            self.research_db.persist()
        elif index_type == 'documents':
            self.documents_db.persist()
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
