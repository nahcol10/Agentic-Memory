import os
import uuid
import json
from dotenv import load_dotenv
from mistralai import Mistral
from traceback import format_exc
from .sql_manager import SQLManager
from .user_manager import UserManager
from .chat_history_manager import ChatHistoryManager
from .search_manager import SearchManager
from .prepare_system_prompt import prepare_system_prompt_for_agentic_chatbot_v3
from .utilities import Utilities
from .load_config import LoadConfig
from .vectordb_manager import VectorDBManager

load_dotenv()


class ChatBot:
    """
    Chatbot class that handles conversational flow, manages user data, and executes function calls using OpenAI's API.
    """

    def __init__(self):
        """
        Initializes the Chatbot instance.

        Sets up OpenAI client, configuration settings, session ID, and database managers.
        """
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        self.cfg = LoadConfig()
        self.chat_model = self.cfg.chat_model
        self.summary_model = self.cfg.summary_model
        self.temperature = self.cfg.temperature
        self.max_history_pairs = self.cfg.max_history_pairs

        self.session_id = str(uuid.uuid4())
        self.utils = Utilities()
        self.sql_manager = SQLManager(str(self.cfg.db_path))
        self.user_manager = UserManager(self.sql_manager)
        self.chat_history_manager = ChatHistoryManager(
            self.sql_manager, str(self.user_manager.user_id) if self.user_manager.user_id else "", self.session_id, self.client, self.summary_model, self.cfg.max_tokens)

        self.vector_db_manager = VectorDBManager(self.cfg)

        self.search_manager = SearchManager(
            self.sql_manager, self.utils, self.client, self.summary_model, self.cfg.max_characters)
        self.agent_functions = [self.utils.jsonschema(self.user_manager.add_user_info_to_database),
                                self.utils.jsonschema(self.vector_db_manager.search_vector_db)]

    def execute_function_call(self, function_name: str, function_args: dict) -> tuple[str, str]:
        """
        Executes the requested function based on the function name and arguments.

        Args:
            function_name (str): The name of the function to execute.
            function_args (dict): The arguments required for the function.

        Returns:
            tuple[str, str]: A tuple containing the function state and result.
        """
        try:
            if function_name == "search_vector_db":
                return self.vector_db_manager.search_vector_db(**function_args)
            elif function_name == "add_user_info_to_database":
                return self.user_manager.add_user_info_to_database(function_args)
            else:
                return "Function call failed.", f"Unknown function: {function_name}"
        except Exception as e:
            return "Function call failed.", f"Error executing {function_name}: {str(e)}"

    def chat(self, user_message: str) -> str:
        """
        Handles a conversation with the user, manages chat history, and executes function calls if needed.

        Args:
            user_message (str): The message from the user.

        Returns:
            str: The chatbot's response or an error message.
        """
        function_call_result_section = ""
        function_call_state = None
        chat_state = "thinking"
        function_call_count = 0
        function_name = ""
        function_args = {}
        function_call_result = ""
        
        self.chat_history = self.chat_history_manager.chat_history
        self.previous_summary = self.chat_history_manager.get_latest_summary()
        
        while chat_state != "finished":
            try:
                if function_call_state == "Function call successful.":
                    chat_state = "finished"
                    if function_name == "add_user_info_to_database":
                        self.user_manager.refresh_user_info()
                    function_call_result_section = (
                        f"## Function Call Executed\n\n"
                        f"- The assistant just called the function `{function_name}` in response to the user's most recent message.\n"
                        f"- Arguments provided:\n"
                        + "".join([f"  - {k}: {v}\n" for k,
                                   v in function_args.items()])
                        + f"- Outcome: ✅ {function_call_state}\n\n"
                        "Please proceed with the conversation using the new context.\n\n"
                        + f"{function_call_result}"
                    )
                    print("************************************")
                    print(function_call_result)
                    print("************************************")
                elif function_call_state == "Function call failed.":
                    function_call_result_section = (
                        f"## Function Call Attempted\n\n"
                        f"- The assistant attempted to call `{function_name}` with the following arguments:\n"
                        + "".join([f"  - {k}: {v}\n" for k,
                                   v in function_args.items()])
                        + f"- Outcome: ❌ {function_call_state} - {function_call_result}\n\n"
                        "Please assist the user based on this result."
                    )

                if function_call_count >= self.cfg.max_function_calls:
                    function_call_result_section = f"""  # Function Call Limit Reached.\n
                    Please conclude the conversation with the user based on the available information."""
                
                system_prompt = prepare_system_prompt_for_agentic_chatbot_v3(
                    str(self.user_manager.user_info) if self.user_manager.user_info else "",
                    self.previous_summary or "",
                    str(self.chat_history),
                    function_call_result_section)
                print("\n\n==========================================")
                print(f"System prompt: {system_prompt}")

                print("\n\nchat_State:", chat_state)
                response = self.client.chat.complete(
                    model=self.chat_model,
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": user_message}],
                    temperature=self.cfg.temperature
                )

                if response.choices[0].message.content:
                    assistant_response = response.choices[0].message.content
                    if isinstance(assistant_response, str):
                        self.chat_history_manager.add_to_history(
                            user_message, assistant_response, self.max_history_pairs
                        )
                        self.chat_history_manager.update_chat_summary(
                            self.max_history_pairs
                        )
                        chat_state = "finished"
                        msg_pair = {"user": user_message, "assistant": assistant_response}
                        self.vector_db_manager.update_vector_db(msg_pair)
                        function_call_state = None
                        self.vector_db_manager.refresh_vector_db_client()
                        return assistant_response
                    else:
                        return "Error: Invalid response format from the model."
                else:
                    # For now, return a fallback response since we're not using tools
                    fallback_response = self.client.chat.complete(
                        model=self.chat_model,
                        messages=[{"role": "system", "content": system_prompt},
                                  {"role": "user", "content": user_message}],
                        temperature=self.cfg.temperature
                    )
                    assistant_response = fallback_response.choices[0].message.content
                    if isinstance(assistant_response, str):
                        self.chat_history_manager.add_to_history(
                            user_message, assistant_response, self.max_history_pairs
                        )
                        msg_pair = {"user": user_message, "assistant": assistant_response}
                        self.vector_db_manager.update_vector_db(msg_pair)
                        function_call_state = None
                        self.vector_db_manager.refresh_vector_db_client()
                        return assistant_response
                    else:
                        return "Error: Invalid fallback response format from the model."

            except Exception as e:
                return f"Error: {str(e)}\n{format_exc()}"
        
        return "Error: Chat loop ended unexpectedly."