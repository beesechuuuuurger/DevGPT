
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool
import openai
import os
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.embeddings import OpenAIEmbeddings
from langchain.agents import BaseMultiActionAgent
from langchain.agents import AgentExecutor
from tokens_embedding import TextProcessor
import itertools
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from db_script import LocalIndex
from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env.

openai_api_key = os.getenv('OPENAI_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
google_cse_id = os.getenv('GOOGLE_CSE_ID')

import json

class LearnToolArgs(BaseModel):
    query: str = Field(..., description="The query to learn about")

class LearnTool(BaseTool):
    name: str = "learn_tool"
    args_schema: Type[BaseModel] = LearnToolArgs
    description: str = "Learn about a topic by searching the web and indexing the results."
    tools: dict = Field(default_factory=dict)  # Define tools as a Pydantic Field
    index: Optional[LocalIndex] = 'research_index'  # Declare index attribute

    def __init__(self):
        super().__init__()
        self.tools['search'] = GoogleSearchAPIWrapper() 
        self.tools['text'] = TextProcessor()
        self.tools['embeddings'] = OpenAIEmbeddings()
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        self.tools['text_splitter'] = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.index = LocalIndex(os.getenv("OPENAI_API_KEY"))  # Initialize LocalIndex


    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            print("Running _run...")
            webpage_titles_and_content = self.process_url(query)
            if not webpage_titles_and_content:
                print("No text found in the search results.")
                return "No text found in the search results."
            prepared_data = self.generate_embeddings(webpage_titles_and_content)
            self.upsert_in_batches(prepared_data, 'research_data', batch_size=100)  # Pass 'research' as index_type to upsert_in_batches
            results = self.research_db.similarity_search(query)
            return results
        except Exception as e:
            print(f"Error: {e}")
            return "An error occurred."

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        # Call your existing methods here
        title, text = await self.process_url(query)  
        structured_data = text
        prepared_data = await self.generate_embeddings(structured_data, query)  
        await self.upsert_in_batches(prepared_data, 'research_data', batch_size=100) 
        results = await self.research_db.similarity_search(query)
        return results

    def process_url(self, query):
        print("Running process_url...")
        # Initialize the GoogleSearchAPIWrapper tool


        # Run the GoogleSearchAPIWrapper tool
        results = self.tools['search'].run(query)

        # Parse the results into a dictionary
        try:
            results_dict = json.loads(results)
        except json.JSONDecodeError:
            print("Failed to parse results into a dictionary.")
            return {}

        # Return the results
        return results_dict


    @staticmethod
    def prepare_data_for_chroma(data):
        """
        Prepare data for insertion into Chroma.

        Args:
            data (dict): A dictionary where keys are unique identifiers (IDs) and values are the corresponding vectors.

        Returns:
            dict: A dictionary where keys are IDs and values are the corresponding vectors, ready for insertion into Chroma.
        """
        # Ensure data is a dictionary
        if not isinstance(data, dict):
            raise TypeError("Data must be a dictionary.")

        # Ensure all values in the dictionary are lists (vectors)
        for key, value in data.items():
            if not isinstance(value, list):
                raise TypeError(f"Value associated with key '{key}' is not a list.")

        return data

    def generate_embeddings(self, search_results):
        if not search_results:
            print("No text to generate embeddings from.")
            return {} 
        print("Running generate_embeddings...")

        chroma_vectors = {}
        for title, text in search_results.items():
            sentences = self.tools['text'].split_text(text, 8000)
            embedding = self.tools['embeddings'].embed_documents(sentences)
            chroma_vectors[title] = embedding.tolist()

        # Prepare data for Chroma
        prepared_data = self.prepare_data_for_chroma(chroma_vectors)

        return prepared_data

    @staticmethod
    def chunks(iterable, batch_size=100):
        """A helper function to break an iterable into chunks of size batch_size."""
        it = iter(iterable)
        chunk = tuple(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            chunk = tuple(itertools.islice(it, batch_size))

    def upsert_in_batches(self, vectors, index_type, batch_size=100):
        print("Running upsert_in_batches...")
        for i, batch in enumerate(self.chunks(vectors.items(), batch_size)):
            print(f"Batch {i}: {batch}")
            try:
                docs = {key: self.text_splitter.split_documents(value) for key, value in batch}
                self.index.create_index(docs, index_type)  # Use LocalIndex to create index
                print(f"Upserted batch {i + 1}/{(len(vectors) + batch_size - 1)//batch_size}")
            except Exception as e:
                print(f"Error when upserting batch {i}: {e}")
                print(f"Batch that caused error: {batch}")
                continue  # Skip this batch and continue with the next one