import os
import uuid
from dotenv import load_dotenv
from load_config import LoadConfig
from mistralai import Mistral
from sql_manager import SQLManager
from user_manager import UserManager
from chat_history_manager import ChatHistoryManager
from prepare_system_prompt import prepare_system_prompt

load_dotenv()

class ChatBot:
    """
    Chatbot class that handle conversational flow
    """
    def __init__(self):
        """
        Initialize the chatbot instance

        Setup Mistralai Client, Configuration Setting, Session ID, and database manager
        """
        self.cfg = LoadConfig()
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        self.chat_model = self.cfg.chat_model
        self.summary_model = self.cfg.summary_model
        self.max_history_pairs = self.cfg.max_history_pairs

        self.sql_manager = SQLManager(self.cfg.db_path)
        self.user_manager = UserManager(self.sql_manager)
        self.session_id = str(uuid.uuid4())
        self.chat_history_manager = ChatHistoryManager(self.sql_manager,self.user_manager.user_id,self.session_id,self.client,self.summary_model,self.cfg.max_tokens)

    def chat(self, user_message: str) -> str:
        """
        Handles the conversation with user and manages chat history
        :param user_message: The message from user
        :return: The Chatbot response or an error
        """
        self.previous_summary = self.chat_history_manager.get_latest_summary()
        system_prompt = prepare_system_prompt(
            self.user_manager.user_info,
            self.previous_summary,
            self.chat_history_manager.chat_history
        )
        print("System Prompt: ",system_prompt)

        try:
            response = self.client.chat.complete(
                model = self.chat_model,
                messages=[
                    {"role":"system","content":system_prompt},
                    {"role":"user","content":user_message}
                ]
            )
            assistance_response = response.choices[0].message.content
            self.chat_history_manager.add_to_history(
                user_message,assistance_response,self.max_history_pairs
            )
            self.chat_history_manager.update_chat_summary(self.max_history_pairs)
            return assistance_response
        except Exception as e:
            return f"Error: {str(e)}"
