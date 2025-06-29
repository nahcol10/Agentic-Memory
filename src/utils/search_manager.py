from mistralai import Mistral
from .utilities import Utilities
from .sql_manager import SQLManager

class SearchManager:
    def __init__(self, sql_manager: SQLManager, utils: Utilities, client: Mistral, summary_model: str, max_characters: int = 1000):
        """
        Initializes the SearchManager instance
        :param sql_manager: The database manager instance
        :param utils: The utility class instance
        :param client: The mistral client instance
        :param summary_model: The summary model to use
        :param max_characters: The maximum number of chatacter to summarize
        """
        self.sql_manager = sql_manager
        self.utils = utils
        self.client = client
        self.summary_model = summary_model
        self.max_characters = max_characters

    def search_chat_history(self,search_term: str) -> tuple[str, str]:
        """
        Searches chat history for a term, performing a case insensitive lookup.
        :param search_term: The keyword to search in chat_history
        :return: tuple containing function state and result
        """
        try:
            search_term = search_term.lower()
            query = """
            SELECT question, answer, timestamp FROM chat_history
            WHERE lower(question) LIKE ? OR LOWER(answer) LIKE ?
            ORDER BY timestamp ASC
            LIMIT 3;                
            """

            results = self.sql_manager.execute_query(query,(f"%{search_term}%",f"%{search_term}%"),fetch_all=True)
            #Ensure the results maintain the order of questions, then answer
            formatted_result = [(q, a, t) for q, a, t in results]
            if formatted_result == []:
                return "Function call failed.", "No result found. Please Try again with different word."

            num_of_characters = self.utils.count_number_of_character(str(results))
            print(f"Number of characters in search results : {num_of_characters}")

            if num_of_characters > self.max_characters:
                result_summary = self.summarize_search_result(str(formatted_result))
                return "Function call successful.", result_summary
            return "Function call successful.", str(formatted_result)
        except Exception as e:
            return "Function call failed.",f"Error : {e}"

    def summarize_search_result(self, search_result: str) -> str:
        """
        Summarize the search result if it exceeds the character limit
        :param search_result: The search result to summarize
        :return: A summarized version of search results
        """
        response = self.client.chat.complete(
            model=self.summary_model,
            messages=[
                {"role": "system", "content": f"Summarize the following content within {self.max_characters} characters"},
                {"role": "user", "content": search_result}
            ]
        )
        response = response.choices[0].message.content
        return response
