from typing import Optional, Type, Dict, List
from pydantic import BaseModel, Field
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool
from langchain.tools.file_management.utils import (
    INVALID_PATH_TEMPLATE,
    BaseFileToolMixin,
    FileValidationError,
)
from langchain.document_loaders import DirectoryLoader
from embeddings import EmbeddingProcessor

import os

class ReadAllFilesInDirectoryInput(BaseModel):
    """Input for ReadAllFilesInDirectoryTool."""
    directory_path: str = Field(..., description="Directory path")

class ReadAllFilesInDirectoryTool(BaseFileToolMixin, BaseTool):
    name: str = "read_all_files_in_directory"
    args_schema: Type[BaseModel] = ReadAllFilesInDirectoryInput
    description: str = "Read all files in a directory from disk"
    embedding_processor: EmbeddingProcessor = Field(default_factory=EmbeddingProcessor)

    def __init__(self):
        super().__init__()
        self.embedding_processor = EmbeddingProcessor()

    def _run(
        self,
        directory_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, str]:
        contents = {}
        try:
            # Use the DirectoryLoader to load the documents in the directory
            loader = DirectoryLoader(directory_path)
            documents = loader.load()

            # Use the EmbeddingProcessor to process the documents
            for document in documents:
                embeddings = self.embedding_processor.generate_embeddings(document)
                # Store the embeddings in the contents dictionary
                contents[document] = embeddings

        except Exception as e:
            contents[directory_path] = "Error: " + str(e)
        return contents

    async def _arun(
        self,
        directory_path: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, str]:
        # TODO: Add aiofiles method
        raise NotImplementedError
