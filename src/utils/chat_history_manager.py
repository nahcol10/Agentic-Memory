from importlib.resources import contents
from typing import Optional, List
from mistralai import Mistral
from .sql_manager import SQLManager
from .utilities import Utilities
import json

class ChatHistoryManager:
    """
    Manages chat history and summarization for a user session
    """

    def __init__(self,sql_manager: SQLManager,user_id: str,session_id: str, client: Mistral, summary_model: str,max_tokens: int) -> None:
        self.utils = Utilities()
        self.client = client
        self.summary_model = summary_model
        self.max_tokens = max_tokens
        self.sql_manager = sql_manager
        self.user_id = user_id
        self.session_id = session_id
        self.chat_history = []
        self.pairs_since_last_summary = 0 #track pair added since last summary

    def add_to_history(self,user_message: str,assistant_response: str, max_history_pairs: int) -> None:
        """
        Add the user message and assistant response to the chat history and save to the database.
        :param user_message: The user's message
        :param assistant_response: The assistant's response
        :param max_history_pairs: The maximum number of message pairs to keep in history.
        :return: None
        """
        self.chat_history.append({"user": user_message})
        self.chat_history.append({"assistant": assistant_response})

        if len(self.chat_history) > max_history_pairs * 2:
            self.chat_history = self.chat_history[-max_history_pairs*2:]
        self.save_to_db(user_message,assistant_response)
        self.pairs_since_last_summary += 1
        print("Chat history saved to database. ")
        chat_history_token_count = self.utils.count_number_of_tokens(str(self.chat_history))
        if chat_history_token_count > self.max_tokens:
            print("Summarizing the Chat History... ")
            print("\n Old number of tokens : ",chat_history_token_count)

            self.summarize_chat_history()
            chat_history_token_count = self.utils.count_number_of_tokens(str(self.chat_history))
            print("\n New number of tokens : ",chat_history_token_count)

    def save_to_db(self,user_message:str, assistant_response:str) -> None:
        """
        Saves the user_message and assistant_response to databases.
        :param user_message: The user's message
        :param assistant_response: The assistant's response
        :return: None
        """
        if not self.user_id:
            print("No user found in database.")
            return

        query = """
            INSERT INTO chat_history (user_id, question, answer, session_id)
            VALUES (?, ?, ?, ?);
        """
        self.sql_manager.execute_query(query,(self.user_id,user_message,assistant_response,self.session_id))

    def get_latest_chat_pairs(self,num_pairs) -> List[tuple]:
        """
        Fetches the latest `num_pairs` user-assistant pair form database

        :param num_pairs: number of pairs to retrieve
        :return:
            List[tuple]: List of tuple containing user question and assistant answer
        """
        query = """
            SELECT question,answer from chat_history
            where session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?;
        """
        chat_data = self.sql_manager.execute_query(query,(self.session_id,num_pairs * 2),fetch_all=True)
        #reverse to maintain the chronological order
        return list(reversed(chat_data))

    def get_latest_summary(self) -> Optional[str]:
        """
        Retrieves the latest summary from the current session from the database.
        :return:
            Optional[str]: The latest summary or None if no summary exists
        """
        query = """
            SELECT summary_text FROM summary
            WHERE session_id = ? ORDER BY timestamp DESC LIMIT 1;
        """
        summary = self.sql_manager.execute_query(query,(self.session_id,),fetch_one=True)
        return summary[0] if summary else None

    def save_summary_to_db(self,summary_text:str) -> None:
        """
        Saves a generated summary to database.

        :param summary_text: The summary text to save.
        :return: None
        """
        if not self.user_id and summary_text:
            return
        query = """
            INSERT INTO summary (user_id, session_id, summary_text) VALUES (?,?,?);
        """
        self.sql_manager.execute_query(query,(self.user_id,self.session_id,summary_text))
        print("Summary saved to database!!")

    def update_chat_summary(self, max_history_pairs :int) -> None:
        """
        Update the chat summary when {max_history_pairs} new pairs have been added since the last summary.

        :param max_history_pairs: Number of pairs required to trigger the summary generation.
        :return: None
        """
        print("Pair since last summary: ",self.pairs_since_last_summary)
        if self.pairs_since_last_summary < max_history_pairs:
            return None
        #Fetch the latest two pairs (if available)
        chat_data = self.get_latest_chat_pairs(max_history_pairs)

        #Fetch the previous summary
        previous_summary = self.get_latest_summary()

        #only generate the summary if there is only two pairs
        if len(chat_data) <= max_history_pairs:
            print("Insufficient Chat data. Skip summary.")
            return

        summary_text = self.generate_the_new_summary(
            self.client,self.summary_model,chat_data,previous_summary
        )

        if summary_text:
            self.save_summary_to_db(summary_text)
            self.pairs_since_last_summary = 0
            print("Chat history summary generated and saved to database.")

    def generate_the_new_summary(self,client: Mistral, summary_model: str, chat_data: List[tuple], previous_summary: Optional[str]) -> Optional[str]:
        """
        Generate the summary from the latest two pairs and previous summary
        :param client: The Client Object used for calling AI model
        :param summary_model: The model name to user for summarization
        :param chat_data: A list of tuples containing user question and assistant answer
        :param previous_summary: The previous summary, if available
        :return:
            Optional[str]: The generated summary or None if an error occurs.
        """
        if not chat_data:
            return None

        #start building the prompt
        summary_prompt = "Summarize the following converstation: \n\n"

        if previous_summary:
            summary_prompt += f"Previous summary :\n {previous_summary} \n\n"

        for q,a in chat_data:
            summary_prompt += f"User: {q}\n Assistant: {a} \n\n"

        summary_prompt += "Provide consice summary while preserving the imporatant details."

        try:
            response = client.chat.complete(
                model=summary_model,
                messages=[
                    {"role": "system","content": summary_prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting summary : {str(e)}")
            return None

    def summarize_chat_history(self):
        """
        Summarize older part of chat history to reduce token count while maintaining context
        :return:
        """
        #select older pair to summarize (keep latest pair untouched)
        pairs_to_keep = 1
        pairs_to_summarize = self.chat_history[:-pairs_to_keep * 2]

        if len(pairs_to_summarize) == 0:
            return

        #Create prompt for summarization
        prompt = f"""
        Summarize the following conversation  while preserving the key details and the conversational's tone:
        {pairs_to_summarize}
        Return the summarized conversation (in JSON format with 'user' and 'assistant' pairs)
        """
        try:
            #Use gpt model to generate the summary
            response = self.client.chat.complete(
                model=self.summary_model,
                messages=[
                    {"role":"user","content":prompt}
                ],
                max_tokens=300
            )
            summarized_pairs = response.choices[0].message.content
            summarized_pairs = json.loads(summarized_pairs)

            #If output is single dictionary warp it in a list
            if isinstance(summarized_pairs,dict):
                summarized_pairs = [summarized_pairs]

            #Ensure the format is consistent
            # ✅ Ensure the format is consistent
            if isinstance(summarized_pairs, list) and all(
                    isinstance(pair, dict) and 'user' in pair and 'assistant' in pair
                    for pair in summarized_pairs
            ):
                # Keep latest pairs + summarized history
                self.chat_history = summarized_pairs + \
                                    self.chat_history[-pairs_to_keep * 2:]
                print("Chat history summarized.")
            else:
                raise ValueError("Invalid format received from LLM.")
        except Exception as e:
            print(f"Failed to summarize chat history: {e}")





