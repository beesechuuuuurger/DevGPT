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
from tokens_embedding import EmbeddingProcessor, TextProcessor
import itertools
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
class LearnToolArgs(BaseModel):
    query: str = Field(..., description="The query to learn about")


class LearnTool(BaseTool):
    name: str = "learn_tool"
    args_schema: Type[BaseModel] = LearnToolArgs
    description: str = "Learn about a topic by searching the web and indexing the results."
    tools: dict = Field(default_factory=dict)  # Define tools as a Pydantic Field
    persist_directory: str  = "D:\\DEV\\codeGPT\\codegpt\\vectorstore\\research_index" # Add this line

    def __init__(self):
        super().__init__()
        self.tools['search'] = GoogleSearchAPIWrapper()
        self.tools['text'] = TextProcessor()
        self.tools['embeddings'] = OpenAIEmbeddings()
        os.environ["GOOGLE_CSE_ID"] = "c1563a217961c415f"
        os.environ["GOOGLE_API_KEY"] = "AIzaSyBcK4qjqXfhHmZoJnSdMizKqXC2z51Zr5g"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.tools['text_splitter'] = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.persist_directory = 'D:\\DEV\\codeGPT\\codegpt\\vectorstore\\research_index'
        self.tools['index'] = Chroma(persist_directory=self.persist_directory, embedding_function=self.tools(['embeddings'].embed_documents))

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            print("Running _run...")
            title, text = self.process_url(query)
            if not text:
                print("No text found in the search results.")
                text = ""
            structured_data = text
            print(f"Structured data: {structured_data[:100]}...")  # Print the first 100 characters of the structured data
            prepared_data = self.generate_embeddings(structured_data, query)
            print(f"Pinecone vectors: {prepared_data}")
            self.upsert_in_batches(prepared_data, batch_size=100)
            results = self.tools['index'].query(query, top_k=10)
            print(f"Results: {results}")
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
        title, text = await self.process_url(query)  # This is not an async function
        structured_data = text  # If you want to use the text part of the tuple
        prepared_data = await self.generate_embeddings(structured_data, query)  # This is not an async function
        await self.upsert_in_batches(prepared_data, batch_size=100) 


    def process_url(self, query):
        print("Running process_url...")
        # Run the GoogleSearchAPIWrapper tool
        results = self.tools['search'].run({"q": query, "k": 10})
        print(f"Search results: ",(results))

        # Check if results is a string
        if isinstance(results, str):
            # Try to parse results into a dictionary
            try:
                import json
                results = json.loads(results)
            except json.JSONDecodeError:
                print("Failed to parse results into a dictionary.")
                return None, None

        # Check if the results contain the expected 'results' field
        if 'results' not in results:
            print("Search results do not contain the expected 'results' field.")
            return None, None

        # Initialize empty lists for titles and texts
        titles = []
        texts = []

        # Iterate over each result
        for result in results['results']:
            # Check if the result has the 'title' and 'full_content' fields
            if 'title' in result and 'full_content' in result:
                # Append the title and full_content to the respective lists
                titles.append(result['title'])
                texts.append(result['full_content'])

        # Return the titles and texts
        return titles, texts

    def generate_embeddings(self, titles, texts):
        if not texts:
            print("No text to generate embeddings from.")
            return [] 
        print("Running generate_embeddings...")
        sentences = [self.tools['text'].split_text(text, 8000) for text in texts]
        print(f"Sentences: {sentences[:5]}...")  # Print the first 5 sentences

        embeddings = [self.tools['embeddings'].generate_embeddings(sentence) for sentence in sentences]
        print(f"Embeddings: {embeddings}")

        # Prepare data for Pinecone
        pinecone_vectors = {title: embedding.tolist() for title, embedding in zip(titles, embeddings)}
        prepared_data = self.prepare_data_for_pinecone(pinecone_vectors)

        print(f"Generated embeddings: {prepared_data}")

        return prepared_data



    @staticmethod
    def prepare_data_for_pinecone(data):
        """
        Prepare data for insertion into Pinecone.

        Args:
            data (dict): A dictionary where keys are unique identifiers (IDs) and values are the corresponding vectors.

        Returns:
            dict: A dictionary where keys are IDs and values are the corresponding vectors, ready for insertion into Pinecone.
        """
        # Ensure Pinecone is installed
        try:
            import pinecone
        except ImportError:
            raise ImportError("Pinecone module is not installed. Install it using 'pip install pinecone-client'.")

        # Ensure data is a dictionary
        if not isinstance(data, dict):
            raise TypeError("Data must be a dictionary.")

        # Ensure all values in the dictionary are lists (vectors)
        for key, value in data.items():
            if not isinstance(value, list):
                raise TypeError(f"Value associated with key '{key}' is not a list.")

        # Convert lists to Pinecone vectors
        for key, value in data.items():
            data[key] = pinecone.Vector(value)

        return data


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
        for i, batch in enumerate(self.chunks(vectors, batch_size)):
            print(f"Batch {i}: {batch}")
            try:
                docs = self.text_splitter.split_documents(batch)
                self.tools['index'] = Chroma.from_documents(docs, self.embedding, persist_directory=self.persist_directory)
                self.tools['index'].persist()
                print(f"Upserted batch {i + 1}/{(len(vectors) + batch_size - 1)//batch_size}")
            except Exception as e:
                print(f"Error when upserting batch {i}: {e}")
                print(f"Batch that caused error: {batch}")
                continue  # Skip this batch and continue with the next one

class IndexQueryToolArgs(BaseModel):
    index_name: str = Field(..., description="The name of the index to query")
    query: str = Field(..., description="The query to use")

class IndexQueryTool(BaseTool):
    name: str = "pinecone_query_tool"
    args_schema: Type[BaseModel] = IndexQueryToolArgs
    description: str = "Query a Pinecone index."
    tools: dict = Field(default_factory=dict)
    persist_directory: str  = "D:\\DEV\\codeGPT\\codegpt\\vectorstore\\research_index" 

    def __init__(self):
        super().__init__()
        self.tools['text'] = TextProcessor()
        self.tools['embeddings'] = OpenAIEmbeddings()
        self.persist_directory = 'D:\\DEV\\codeGPT\\codegpt\\vectorstore\\research_index'
        self.tools['index'] = Chroma(persist_directory=self.persist_directory, embedding_function=self.tools['embeddings'].embed_documents)

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
        query_action = AgentExecutor(tool="Index_Query_Tool", input={"index_name": "codegpt", "query": input})
        query_result = self.run_action(query_action)

        return learn_tool_result, query_result
