from typing import List, Union, Dict
import pinecone 
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.tools import ShellTool
from langchain.chains import LLMChain
from tempfile import TemporaryDirectory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes.vectorstore import VectorstoreIndexCreator 
from langchain.chains import ConversationChain
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
from learn import LearnTool, PineconeQueryTool, LearnAndQueryAgent
import json
from tokens_embedding import EmbeddingProcessor, TextProcessor

# Initialize the processors
embedding_processor = EmbeddingProcessor()
text_processor = TextProcessor()

embeddings = OpenAIEmbeddings()
PINECONE_API_KEY = "068f95dc-2a64-4665-9d8d-2aa67205f07e"
PINECONE_ENV = "asia-southeast1-gcp"

index_name = "codegpt"
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index("codegpt")
# Create a temporary directory for the file tools
working_directory = TemporaryDirectory()

llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

search = GoogleSearchAPIWrapper()
python_repl = PythonREPL()
write_file = WriteFileTool()
read_file = ReadFileTool()
list_directory = ListDirectoryTool()
copy_file = CopyFileTool()
file_delete = DeleteFileTool()
file_search = FileSearchTool()
move_file = MoveFileTool()
shell_tool = ShellTool()
learn_tool = LearnTool()
pinecone_query_tool = PineconeQueryTool

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
        name="Pinecone_Query_Tool",
        func=pinecone_query_tool.run,
        description="Query the Pinecone Index for more context."
    ),
    Tool(
        name="Python_repl",
        description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
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
        description="Create a copy of a file in a specified location."
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
]

template = """You are an AI assistant that can help with research, coding, and general chat. You have access to the following tools:

{tools}

Tool Commands: 
Google Search: "search", args: "input": "<search>"
Write to file: "write_file", "<file>", "<text>"
Read file: "read_file", "<file>"
Delete file: "file_delete", "<file>"
Search Files: "file_search", args: "directory": "<directory>"
Evaluate Code: "Python_repl", args: "code": "<full_code_string>"
Get Improved Code: "improve_code", args: "suggestions": "<list_of_suggestions>", "code": "<full_code_string>"
User Assistance: "Attention Needed", args: "reason": "<reason>"

Here is some context from a vector storage of the users input:
{context}

Use the following format:

Input: the users input
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: how can the results of this action effect the rest of the process
Final Answer: the final answer to the original input question

Begin! 

Input: {text}
{agent_scratchpad}"""

# Set up a prompt template
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
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        # Get the context from the Pinecone index
        context = index.query(kwargs["input"])
        # Set the context variable to that value
        kwargs["context"] = context
        formatted = self.template.format(**kwargs)

        # Initialize a CharacterTextSplitter with the model's maximum context length
        splitter = CharacterTextSplitter(max_length=4097)

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

if __name__ == "__main__":

    agent_chain = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, memory=memory)

    while True:
        try:
            user_input = input("Enter a command (type 'exit' to quit): ")
            if user_input.lower() == 'exit':
                break
            response = agent_chain.run(input={"input": user_input})
            if response["action"] == "Learn_Tool":
                vectors_filename = response["observation"]
                with open(vectors_filename, 'r') as f:
                    vectors = json.load(f)
                print(f"Learned about {len(vectors)} topics related to {user_input}.")
            else:
                print(response)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")
