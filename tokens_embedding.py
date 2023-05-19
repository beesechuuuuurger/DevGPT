import tiktoken
from langchain.embeddings import OpenAIEmbeddings
import numpy as np

class EmbeddingProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()

    def generate_embeddings(self, documents):
        embeddings = []
        for text in documents:
            embedding = self.embeddings.embed_query(text)  # Assume 'embed_query' is the correct method
            embeddings.append(embedding)
        return embeddings


    def split_text(self, text, max_length):
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        for word in words:
            current_length += len(word) + 1
            if current_length > max_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = len(word) + 1
            current_chunk.append(word)
        chunks.append(' '.join(current_chunk))
        return chunks

class TextProcessor:
    def __init__(self):
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def count_tokens(self, text):
        tokens = self.encoding.encode(text)
        return len(tokens)

    def split_text(self, text, max_tokens):
        max_tokens = 4000  # Adjust max_tokens
        words = text.split()
        chunks = []
        current_chunk = []
        current_tokens = 0
        for word in words:
            word_tokens = self.count_tokens(word)
            current_tokens += word_tokens
            if current_tokens > max_tokens:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_tokens = word_tokens
            current_chunk.append(word)
        chunks.append(' '.join(current_chunk))
        return chunks

