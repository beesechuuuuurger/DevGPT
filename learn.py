from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool
import pinecone
import json
import spacy
import numpy as np
import openai
from datetime import datetime
import os
from langchain.utilities import GoogleSearchAPIWrapper
import time
from langchain.embeddings import OpenAIEmbeddings
from langchain.agents import BaseMultiActionAgent
from langchain.agents import AgentExecutor
from tokens_embedding import EmbeddingProcessor, TextProcessor
import itertools

# Initialize the processors
embedding_processor = EmbeddingProcessor()
text_processor = TextProcessor()

class LearnToolArgs(BaseModel):
    query: str = Field(..., description="The query to learn about")


class LearnTool(BaseTool):
    name: str = "learn_tool"
    args_schema: Type[BaseModel] = LearnToolArgs
    description: str = "Learn about a topic by searching the web and indexing the results."
    tools: dict = Field(default_factory=dict)  # Define tools as a Pydantic Field

    def __init__(self):
        super().__init__()
        self.tools['search'] = GoogleSearchAPIWrapper(k=10)
        self.tools['embeddings'] = embedding_processor
        os.environ["GOOGLE_CSE_ID"] = "cse-id"
        os.environ["GOOGLE_API_KEY"] = "api-key"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        pinecone_api = "api-key"
        pinecone_env = "env"
        pinecone.init(api_key=pinecone_api, environment=pinecone_env)
        self.tools['index'] = pinecone.Index(index_name="codegpt")
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        print("Running _run...")
        title, text = self.process_url(query)
        structured_data = text
        print(f"Structured data: {structured_data[:100]}...")  # Print the first 100 characters of the structured data
        pinecone_vectors = self.generate_embeddings(structured_data, query)
        print(f"Pinecone vectors: {pinecone_vectors}")
        self.upsert_in_batches(pinecone_vectors, batch_size=100)
        results = self.tools['index'].query(query, top_k=10)
        print(f"Results: {results}")
        return results

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        # Call your existing methods here
        title, text = self.process_url(query)  # This is not an async function
        structured_data = text  # If you want to use the text part of the tuple
        pinecone_vectors = self.generate_embeddings(structured_data, query)  # This is not an async function
        self.upsert_in_batches(pinecone_vectors, batch_size=100)  # Specify batch size if different from default
        # Query the Pinecone index instead of opening the JSON file
        results = self.tools['index'].query(query, top_k=10)
        return results


    def process_url(self, query):
        print("Running process_url...")
        results = self.tools['search'].run(query)
        print(f"Search results: {results[:100]}...")  # Print the first 100 characters of the search results
        sentences = results.split('. ')
        text = ' '.join(sentences[:1000])
        if not text.strip():
            return None, None
        title = query
        print(f"Title: {title}, Text: {text[:100]}...")  # Print the first 100 characters of the text
        return title, text

    def generate_embeddings(self, content, query):
        print("Running generate_embeddings...")
        chunks = self.tools['embeddings'].split_text(content, 8000)
        print(f"Chunks: {chunks[:5]}...")  # Print the first 5 chunks
        embeddings = []
        pinecone_vectors = []
        for i, chunk in enumerate(chunks):
            embedding = self.tools['embeddings'].generate_embeddings(chunk)
            print(f"Embedding for chunk {i}: {embedding}")
            vector = embedding
            embeddings.append(vector)
        mean_vector = np.mean(embeddings, axis=0).tolist()
        pinecone_vectors.append((str(i), mean_vector, {"title": query[:100]}))
        print(f"Generated embeddings: {pinecone_vectors}")
        return pinecone_vectors

    @staticmethod
    def chunks(iterable, batch_size=100):
        """A helper function to break an iterable into chunks of size batch_size."""
        it = iter(iterable)
        chunk = tuple(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            chunk = tuple(itertools.islice(it, batch_size))


    def upsert_in_batches(self, vectors, batch_size=100):
        print("Running upsert_in_batches...")
        for i, batch in enumerate(LearnTool.chunks(vectors, batch_size)):
            print(f"Batch {i}: {batch}")
            vectors_dict = {id: vector for id, vector, metadata in batch}
            self.tools['index'].upsert(vectors=vectors_dict)
            print(f"Upserted batch {i + 1}/{(len(vectors) + batch_size - 1)//batch_size}")



class PineconeQueryToolArgs(BaseModel):
    index_name: str = Field(..., description="The name of the Pinecone index to query")
    query: str = Field(..., description="The query to use")

class PineconeQueryTool(BaseTool):
    name: str = "pinecone_query_tool"
    args_schema: Type[BaseModel] = PineconeQueryToolArgs
    description: str = "Query a Pinecone index."
    tools: dict = Field(default_factory=dict)

    def __init__(self):
        super().__init__()
        self.tools['index'] = pinecone.Index(index_name="codegpt")

    def _run(
        self,
        index_name: str,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        # Query the Pinecone index
        results = self.tools['index'].query(query, top_k=10)
        return results

class LearnAndQueryAgent(BaseMultiActionAgent):
    def run(self, input):
        # Run the LearnTool
        learn_tool_action = AgentExecutor(tool="Learn_Tool", input=input)
        learn_tool_result = self.run_action(learn_tool_action)

        # Query the Pinecone index
        query_action = AgentExecutor(tool="Pinecone_Query_Tool", input={"index_name": "codegpt", "query": input})
        query_result = self.run_action(query_action)

        return learn_tool_result, query_result
