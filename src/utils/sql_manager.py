import sqlite3

class SQLManager:
    """
    A manager for Handling SQLite database connections and executing queries
    """
    def __init__(self,db_path:str):
        """
        Initialize the SQLManager instance

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path

    def execute_query(self,query:str,params: tuple = (), fetch_one:bool = False, fetch_all:bool = False) -> list:
        """
        Execute SQL query with optional parameters and fetch options
        :param query: The SQL query to execute
        :param params: Parameter to pass to SQL query. Default to ()
        :param fetch_one: Whether to fetch a single row. Default to False
        :param fetch_all: Whether to fetch all rows. Default to False
        :return:
            Optional[list[tuple[Any, ...]]]:
                - A single row (if `fetch_one` is True)
                - All rows (if `fetch_all` is True)
                - None if No data is fetched.
        """

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(query,params)
        result = cursor.fetchone() if fetch_one else cursor.fetchall() if fetch_all else None
        conn.commit()
        conn.close()
        return result

# if __name__ == '__main__':
#     from load_config import LoadConfig
#     cfg = LoadConfig()
#     sql_manager = SQLManager(cfg.db_path)
#     response = sql_manager.execute_query(query="select * from users;",fetch_one=True)
#     print(response)


