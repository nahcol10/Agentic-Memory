import math
from typing import Optional,Dict,Any,Tuple
from .sql_manager import SQLManager

class UserManager:
    """Manages users related operations, including retrieving user information and user ID from the database"""

    def __init__(self,sql_manager:SQLManager):
        """
        Initialize the UserManager with database Manager
        :params sql_manager: The Database manager instance to execute queries.
        """
        self.sql_manager = sql_manager
        self.user_info = self.get_user_info()
        self.user_id = self.get_user_id()

    def get_user_info(self) -> Optional[Dict[str,Any]]:
        """
        Retrieves user information from databases, filtering out empty values, None, and NaN.
        :return:
            Optional[Dict[str, Any]]: A dictionary containing user information with valid values or None if no user
            is found.
        """
        query = "SELECT * FROM user_info LIMIT 10;"
        user = self.sql_manager.execute_query(query,fetch_one=True)
        if user:
            user_info = {
                "id": user[0],
                "name": user[1],
                "last_name": user[2],
                "occupation": user[3],
                "location": user[4],
                "gender": user[5],
                "age": user[6],
                "interests": user[7]
            }
            return {k: v for k, v in user_info.items() if v not in (None, "") and not (isinstance(v, float) and math.isnan(v))}
        return None

    def refresh_user_info(self):
        self.user_info = self.get_user_info()

    def get_user_id(self) -> Optional[int]:
        """
        Retrieves the user_id from the database.
        :return:
            Optional[int]: The User id if found, otherwise None
        """
        query = "SELECT id FROM user_info LIMIT 1;"
        user = self.sql_manager.execute_query(query, fetch_one=True)
        return user[0] if user else None

    def add_user_info_to_database(self, user_info: dict) -> Tuple[str, str]:
        """
        Update the user information in the database if valid keys are provided.
        Merges interest instead of overriding them.

        :param user_info: Dictionary containing user attributes to update.

         Examples:
        >>> add_user_info_to_database({'interests': ['watching movies', 'talking']})
        'User information updated successfully.'

        >>> add_user_info_to_database({'location': 'Italy'})
        'User information updated successfully.'

        :return:
        """
        print("Entering the add user info function:")
        print("user_info: ",user_info)
        print(type(user_info))
        try:
            valid_keys = {
                "name","last_name","age","gender","location","occupation","interests"
            }
            for key in user_info.keys():
                if key not in valid_keys:
                    return "Function call failed","Please provide a valid key from the following list: name, last_name, age, gender, location, occupation, interests"

            processed_info = user_info.copy()

            # Handle merging "interests" if present
            if "interests" in processed_info:
                new_interests = []
                if isinstance(processed_info["interests"], list):
                    new_interests = [i.strip() for i in processed_info["interests"] if isinstance(i, str)]
                elif isinstance(processed_info["interests"], str):
                    new_interests = [i.strip() for i in processed_info["interests"].split(",")]

                query = "SELECT interests FROM user_info LIMIT 1;"
                result = self.sql_manager.execute_query(query, fetch_one=True)
                existing_interests = []
                if result and result[0] and result[0][0]:
                    existing_interests = [i.strip() for i in result[0][0].split(",") if i.strip()]
                merged_interests = sorted(set(existing_interests + new_interests))
                processed_info["interests"] = ", ".join(merged_interests)

            # Prepare SQL SET clause
            set_clause = ", ".join(f"{key} = ?" for key in processed_info.keys())
            params = tuple(processed_info.values())

            if not set_clause:
                return "Function call failed", "No valid field to update"

            query = f"""
                UPDATE user_info
                SET {set_clause}
                WHERE id = (SELECT id FROM user_info LIMIT 1);
                """

            self.sql_manager.execute_query(query=query, params=params)
            return "Function call Successful.", "User information updated"
        except Exception as e:
            print(f"Error : {e}")
            return "Function call Failed.",f"Error: {e}"