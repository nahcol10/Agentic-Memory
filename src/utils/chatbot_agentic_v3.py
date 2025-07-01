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
    Chatbot class that handles conversational flow, manages user data, and executes function calls using Mistral's API.
    """

    def __init__(self):
        """
        Initializes the Chatbot instance.

        Sets up Mistral client, configuration settings, session ID, and database managers.
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
                return self.user_manager.add_user_info_to_database(**function_args)
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
        try:
            # Initialize variables
            function_call_result_section = ""
            function_call_state = None
            function_call_count = 0
            function_name = ""
            function_args = {}
            function_call_result = ""
            
            # Get chat history and previous summary
            self.chat_history = self.chat_history_manager.chat_history
            self.previous_summary = self.chat_history_manager.get_latest_summary()
            
            # Main conversation loop
            while function_call_count < self.cfg.max_function_calls:
                # Build function call result section if there was a previous function call
                if function_call_state is not None:
                    function_call_result_section = self._build_function_call_result_section(
                        function_name, function_args, function_call_state, function_call_result
                    )

                    # If function call was successful and it was add_user_info_to_database, refresh user info
                    if function_call_state == "Function call successful." and function_name == "add_user_info_to_database":
                        self.user_manager.refresh_user_info()

                # Check if we've reached the function call limit
                if function_call_count >= self.cfg.max_function_calls:
                    function_call_result_section += (
                        f"\n\n# Function Call Limit Reached.\n"
                        "Please conclude the conversation with the user based on the available information."
                    )

                # Prepare system prompt
                system_prompt = prepare_system_prompt_for_agentic_chatbot_v3(
                    str(self.user_manager.user_info) if self.user_manager.user_info else "",
                    self.previous_summary or "",
                    str(self.chat_history),
                    function_call_result_section
                )

                # Debug output (consider removing in production)
                print(f"\n\n{'=' * 50}")
                print(f"System prompt: {system_prompt}")
                print(f"Function call count: {function_call_count}")

                # Make API call to Mistral
                response = self.client.chat.complete(
                    model=self.chat_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    tools=self.agent_functions,  # type: ignore
                    tool_choice="auto",
                    temperature=self.temperature
                )

                # Handle response with content (regular message)
                if response.choices[0].message.content:
                    assistant_response = response.choices[0].message.content
                    if isinstance(assistant_response, str):
                        self.chat_history_manager.add_to_history(
                            user_message, assistant_response, self.max_history_pairs
                        )
                        self.chat_history_manager.update_chat_summary(self.max_history_pairs)
                        msg_pair = {"user": user_message, "assistant": assistant_response}
                        self.vector_db_manager.update_vector_db(msg_pair)
                        self.vector_db_manager.refresh_vector_db_client()
                        return assistant_response
                    else:
                        return "Error: Invalid response format from the model."

                # Handle function calls
                elif response.choices[0].message.tool_calls:
                    # Check if we've reached the limit
                    if function_call_count >= self.cfg.max_function_calls:
                        print("Function call limit reached, using fallback response...")
                        assistant_response = self._get_fallback_response(system_prompt, user_message)
                        self.chat_history_manager.add_to_history(
                            user_message, assistant_response, self.max_history_pairs
                        )
                        msg_pair = {"user": user_message, "assistant": assistant_response}
                        self.vector_db_manager.update_vector_db(msg_pair)
                        self.vector_db_manager.refresh_vector_db_client()
                        return assistant_response

                    # Execute function call
                    function_call_count += 1
                    tool_call = response.choices[0].message.tool_calls[0]
                    function_name = tool_call.function.name

                    try:
                        if isinstance(tool_call.function.arguments, str):
                            function_args = json.loads(tool_call.function.arguments)
                        else:
                            function_args = tool_call.function.arguments
                    except json.JSONDecodeError as e:
                        function_call_state = "Function call failed."
                        function_call_result = f"Invalid JSON arguments: {str(e)}"
                        continue

                    print(f"Function requested: {function_name}")
                    print(f"Function arguments: {function_args}")

                    function_call_state, function_call_result = self.execute_function_call(
                        function_name, function_args
                    )

                    # Continue the loop to process the function call result
                    continue

                # Handle edge case where there's neither content nor tool calls
                else:
                    return "I apologize, but I didn't generate a proper response. Please try rephrasing your question."

            # If we exit the loop due to max function calls, provide a fallback response
            print("Maximum function calls reached, providing fallback response...")
            system_prompt = prepare_system_prompt_for_agentic_chatbot_v3(
                str(self.user_manager.user_info) if self.user_manager.user_info else "",
                self.previous_summary or "",
                str(self.chat_history),
                function_call_result_section + "\n\n# Function Call Limit Reached.\nPlease conclude the conversation with the user based on the available information."
            )

            assistant_response = self._get_fallback_response(system_prompt, user_message)
            self.chat_history_manager.add_to_history(
                user_message, assistant_response, self.max_history_pairs
            )
            msg_pair = {"user": user_message, "assistant": assistant_response}
            self.vector_db_manager.update_vector_db(msg_pair)
            self.vector_db_manager.refresh_vector_db_client()
            return assistant_response

        except Exception as e:
            error_msg = f"I apologize, but an error occurred while processing your request. Please try again."
            print(f"Error in chat method: {str(e)}\n{format_exc()}")
            return error_msg

    def _build_function_call_result_section(self, function_name: str, function_args: dict,
                                            function_call_state: str, function_call_result: str) -> str:
        """
        Builds the function call result section for the system prompt.

        Args:
            function_name (str): Name of the executed function
            function_args (dict): Arguments passed to the function
            function_call_state (str): State of the function call
            function_call_result (str): Result of the function call

        Returns:
            str: Formatted function call result section
        """
        if function_call_state == "Function call successful.":
            return (
                f"## Function Call Executed\n\n"
                f"- The assistant just called the function `{function_name}` in response to the user's most recent message.\n"
                f"- Arguments provided:\n"
                + "".join([f"  - {k}: {v}\n" for k, v in function_args.items()])
                + f"- Outcome: ✅ {function_call_state}\n\n"
                  "Please proceed with the conversation using this new context."
                + f"\n{function_call_result}"
            )
        else:
            return (
                f"## Function Call Attempted\n\n"
                f"- The assistant attempted to call `{function_name}` with the following arguments:\n"
                + "".join([f"  - {k}: {v}\n" for k, v in function_args.items()])
                + f"- Outcome: ❌ {function_call_state} - {function_call_result}\n\n"
                  "Please assist the user based on this result."
            )

    def _get_fallback_response(self, system_prompt: str, user_message: str) -> str:
        """
        Gets a fallback response when function calls are exhausted or failed.

        Args:
            system_prompt (str): The system prompt to use
            user_message (str): The user's message

        Returns:
            str: The fallback response
        """
        try:
            fallback_response = self.client.chat.complete(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.temperature
            )
            content = fallback_response.choices[0].message.content
            if isinstance(content, str):
                return content
            else:
                return "I apologize, but I'm experiencing technical difficulties. Please try again."
        except Exception as e:
            return f"I apologize, but I'm experiencing technical difficulties. Please try again. Error: {str(e)}"