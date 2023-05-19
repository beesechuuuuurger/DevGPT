import re
from langchain.callbacks.base import BaseCallbackHandler

class ValidateAndCorrectActionCallback(BaseCallbackHandler):
    def __init__(self):
        self.tool_fields = {
            "Write_File": {
                "Action": "Write_File",
                "Action Input": {"file_path": "", "text": ""}
            },
            "Delete_File": {
                "Action": "Delete_File",
                "Action Input": {"file_path": ""}
            },
            "Move_File": {
                "Action": "Move_File",
                "Action Input": {"source_path": "", "destination_path": ""}
            },
            "Copy_File": {
                "Action": "Copy_File",
                "Action Input": {"source_path": "", "destination_path": ""}
            },
            "Shell_Tool": {
                "Action": "Shell_Tool",
                "Action Input": {"command": ""}
            },
            "Python_repl": {
                "Action": "Python_repl",
                "Action Input": {"code": ""}
            },
            "Search": {
                "Action": "Search",
                "Action Input": {"q": ""}
            },
            "Read_File": {
                "Action": "Read_File",
                "Action Input": {"file_path": ""}
            },
            "List_Directory": {
                "Action": "List_Directory",
                "Action Input": {"directory_path": ""}
            },
            "File_Search": {
                "Action": "File_Search",
                "Action Input": {"directory_path": "", "regex_pattern": ""}
            },
            "Read_All_Files_in_Directory": {
                "Action": "Read_All_Files_in_Directory",
                "Action Input": {"directory_path": ""}
            },
        }

    def on_action(self, agent, context):
        action = context['action']
        action_input = context['action_input']
    
        # Validate the action and action input here
        if action not in self.tool_fields:
            raise ValueError(f"Invalid action: {action}")
    
        if not isinstance(action_input, dict):
            raise ValueError(f"Invalid action input: {action_input}")
    
        # Validate that the action input keys match the expected keys for this action
        expected_input_keys = set(self.tool_fields[action]['Action Input'].keys())
        actual_input_keys = set(action_input.keys())
        if expected_input_keys != actual_input_keys:
            raise ValueError(f"Invalid keys for action input: {action_input}. Expected keys: {expected_input_keys}")

        # Check for numbers in the action and action input and remove them
        context['action'] = re.sub(r'\d+', '', action)
        context['action_input'] = {re.sub(r'\d+', '', k): v for k, v in action_input.items()}

    def on_observation(self, agent, context):
        observation = context['observation']

        # Check for numbers in the observation and remove them
        context['observation'] = re.sub(r'\d+', '', observation)
