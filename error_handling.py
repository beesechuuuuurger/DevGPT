from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Action(BaseModel):
    action: str = Field(description="action to take")
    action_input: str = Field(description="input to the action")

# Initialize the PydanticOutputParser with the Action model
parser = PydanticOutputParser(pydantic_object=Action)

# Initialize the RetryWithErrorOutputParser with the PydanticOutputParser and the OpenAI LLM
retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=OpenAI(temperature=0))

def handle_error(bad_response, prompt_value):
    try:
        # Try to parse the response
        result = parser.parse(bad_response)
    except Exception as e:
        # If an error occurs, retry with the RetryWithErrorOutputParser
        result = retry_parser.parse_with_prompt(bad_response, prompt_value)
    return result
