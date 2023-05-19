from langchain.vectorstores import Chroma
from db_script import LocalIndex
from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env.

openai_api_key = os.getenv('OPENAI_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
google_cse_id = os.getenv('GOOGLE_CSE_ID')

class QueryHandler:
    def __init__(self, local_index):
        self.local_index = LocalIndex()

    def query_research_data(self, query, index_type='research_data'):
        # Query the research data index
        # You'll need to implement this method in the LocalIndex class
        results = self.local_index.similarity_search(query, index_type)
        return results

    def query_file_embeddings(self, query, index_type='documents'):
        # Query the file embeddings index
        # You'll need to implement this method in the LocalIndex class
        results = self.local_index.similarity_search(query, index_type)
        return results

    def handle_query(self, query):
        # Determine which index to query based on the user's input
        # This is a placeholder - you'll need to implement this logic
        index_to_query = 'research_data', 'documents'
        if index_to_query == 'research_data':
            results = self.query_research_data(query)
        elif index_to_query == 'documents':
            results = self.query_file_embeddings(query)
        else:
            raise ValueError(f"Unsupported index: {index_to_query}")

        return results
