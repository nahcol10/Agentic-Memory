import os
import sqlite3
from pyprojroot import here

def create_user_info():
    """
    Create a Sqlite database and initialize tables for user information, chat history, and summaries.

    This function:
    - Creates a data directory if it doesn't exist.
    - Establishes a sqlite connection to 'chatbot.db'.
    - Creates the following tables if they don't exist already:
        - `user_info`: Stores user details (e.g. name, occupation, location, etc.)
        - `chat_history`: Records chat interactions with timestamp and session ids.
        - `summary`: Stores summarized chat sessions.
    - Inserts a sample user (`Lochan Paudel`) if no user record exists.

    Tables:
        user_info:
            - id (INTEGER, PRIMARY KEY)
            - name (TEXT, NOT NULL)
            - last_name (TEXT, NOT NULL)
            - occupation (TEXT, NOT NULL)
            - location (TEXT, NOT NULL)
            - age (INTEGER, NULLABLE)
            - gender (TEXT, NULLABLE)
            - interests (TEXT, NULLABLE)

        chat_history:
            - id (INTEGER, PRIMARY KEY)
            - user_id (INTEGER, FOREIGN KEY -> user_info.id)
            - timestamp (DATETIME, DEFAULT CURRENT_TIMESTAMP)
            - question (TEXT, NOT NULL)
            - answer (TEXT, NOT NULL)
            - session_id (TEXT, NOT NULL)

        summary:
            - id (INTEGER, PRIMARY KEY)
            - user_id (INTEGER, FOREIGN KEY -> user_info.id)
            - session_id (TEXT, NOT NULL)
            - summary_text (TEXT, NOT NULL)
            - timestamp (DATETIME, DEFAULT CURRENT_TIMESTAMP)
    """
    # Create data directory if it doesn't exist
    data_dir = here("data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Directory: {data_dir} was created")

    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(here("data/chatbot.db"))
    cursor = conn.cursor()

    # Create Tables
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS user_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            occupation TEXT NOT NULL,
            location TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            interests TEXT
        );

        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            session_id TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES user_info(id)
        );

        CREATE TABLE IF NOT EXISTS summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            session_id TEXT NOT NULL,
            summary_text TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES user_info(id)
        );
    """)

    # Insert Sample User if Not Exists (leaving age, gender, interests empty)
    cursor.execute("""
        INSERT INTO user_info (name, last_name, occupation, location, age, gender, interests)
        SELECT 'Lochan', 'Paudel', 'ML Engineer', 'Nepal', NULL, NULL, NULL
        WHERE NOT EXISTS (SELECT 1 FROM user_info);
    """)

    # Commit changes and close the connection
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_user_info()