import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))




from typing import List, Union, Dict
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.python import PythonREPL
from langchain.tools import ShellTool
from tempfile import TemporaryDirectory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.utilities import PythonREPL, GoogleSearchAPIWrapper
from langchain.prompts import BaseChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.tools.file_management import (
    ReadFileTool,
    CopyFileTool,
    DeleteFileTool,
    MoveFileTool,
    WriteFileTool,
    ListDirectoryTool,
    FileSearchTool,
)
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os
from scripts.learn import LearnTool
from scripts.tokens_embedding import EmbeddingProcessor, TextProcessor
from scripts.query import QueryHandler
import os
from scripts.db_script import LocalIndex
import json
from dotenv import load_dotenv
import os
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from langchain.callbacks.base import BaseCallbackHandler
import re
from scripts.readmultiplefiles import ReadAllFilesInDirectoryTool
from scripts.validation_agent import ValidateAndCorrectActionCallback
load_dotenv()  # take environment variables from .env.

openai_api_key = os.getenv('OPENAI_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
google_cse_id = os.getenv('GOOGLE_CSE_ID')


# Initialize the processors
embedding_processor = EmbeddingProcessor()
text_processor = TextProcessor()

embeddings = OpenAIEmbeddings()

# Create a temporary directory for the file tools
working_directory = TemporaryDirectory()

llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
working_directory = "D:\DEV\codeGPT\codegpt\llmchain\workspace"


# Initialize the tools with the loaded configuration
write_file = WriteFileTool()
file_delete = DeleteFileTool()
move_file = MoveFileTool()
copy_file = CopyFileTool()
shell_tool = ShellTool()
python_repl = PythonREPL()
search = GoogleSearchAPIWrapper()
read_file = ReadFileTool()
list_directory = ListDirectoryTool()
file_search = FileSearchTool()
learn_tool = LearnTool()
local_index = LocalIndex()
read_multiple_files = ReadAllFilesInDirectoryTool()

def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return result

tools = [
    Tool(
        name="Search",
        description="Get an answer for something basic.",
        func=search.run
    ),
    Tool(
        name="Learn_Tool",
        func=learn_tool.run,
        description="Learn about a complex topic by searching the web and indexing the results."
    ),
    Tool(
        name="Python_repl",
        description="A Python shell. Use this to execute python commands. Input should be a valid python command, NOT DOCSTRING. If you want to see the output of a value, you should print it out with `print(...)`. ",
    func=python_repl.run
    ),
    Tool(
        name="Write_File",
        func=write_file.run,
        description="Write file to disk."
    ),
    Tool(
        name="Read_File",
        func=read_file.run,
        description="Read file from disk"
    ),
    Tool(
        name="List_Directory",
        func=list_directory.run,
        description="List files and directories in a specified folder"
    ),
    Tool(
        name="Copy_File",
        func=copy_file.run,
        description="Create a copy of a file to a specified location."
    ),
    Tool(
        name="File_Delete",
        func=file_delete.run,
        description="Delete a file."
    ),
    Tool(
        name="File_Search",
        func=file_search.run,
        description="Recursively search for files in a subdirectory that match the regex pattern."
    ),
    Tool(
        name="Move_File",
        func=move_file.run,
        description="Move or rename a file from one location to another."
    ),
    Tool(
        name="Shell_Tool",
        func=shell_tool.run,
        description=shell_tool.description
    ),
    Tool(
       name="Read_All_Files_in_Directory",
       func=read_multiple_files.run,
       description="Read multiple files from disk"
    ),
]

template = """You are an AI assistant that can help with research, coding, and general chat. You have access to the following tools:

{tools}



Use the following format:

Input: the users input
Thought: you should always think about what to do
Action: the action to take
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: how can the results of this action effect the rest of the process
Final Answer: the final answer to the original input question

Begin! 

Input: {input}

{agent_scratchpad}"""



class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += "Action: " + action.action + "\n"
            thoughts += "Action Input: " + json.dumps(action.action_input) + "\n"
            thoughts += "Observation: " + observation + "\n"
            thoughts += "Thought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        formatted = self.template.format(**kwargs)

        # Initialize a CharacterTextSplitter with the model's maximum context length
        splitter = CharacterTextSplitter(max_length=3700)

        # Split the formatted prompt into chunks that do not exceed the model's maximum context length
        chunks = splitter.split(formatted)

        # Use the last chunk (this will be the largest chunk that does not exceed the model's maximum context length)
        formatted = chunks[-1]

        return [HumanMessage(content=formatted)]




prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)




def run_agent_chain(user_input):
    response = agent_chain.run(user_input)
    return response

if __name__ == "__main__":
    # Initialize the agent with the callback
    validate_action_callback = ValidateAndCorrectActionCallback()
    agent_chain = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, callbacks=[validate_action_callback])

    while True:
        try:
            user_input = input("Enter a command (type 'exit' to quit): ")
            if user_input.lower() == 'exit':
                break
                
            else:
                # Run the agent chain with the user input
                result = run_agent_chain(user_input)
                # The result is already parsed and corrected by the ValidateAndCorrectActionCallback, so we can directly print it
                print(result)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")
