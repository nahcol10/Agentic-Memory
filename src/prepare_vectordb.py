import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from pyprojroot import here
from utils.load_config import LoadConfig

load_dotenv()

def prepare_vectordb():
    """
    Prepare a vector db using Chromadb and MistralAI embeddings.

    This function setups the vector database by:
        - Loading configuration from `LoadConfig` class.
        - Creating the Mistral Embedding function using provided API key and model
        - Creating the vector database directory if it doesn't exists
        - Initializing the Persistent ChromaDB client at the specified directory
        - Creating or retrieving a collection in the vector database with cosine similarity

    Steps:
        1. Load MistralAI API keys and model name from environment and configuration
        2. Create vector database directory if it doesn't exists alread.
        3. Initialize a chromadb client with persistent storage path.
        4. Create or get a existing collection with specified name and embedding functions
        5. Log the Creation and number of items in the collections.
    :return: None
    """
    cfg = LoadConfig()
    mistral_embedding_function = embedding_functions.MistralEmbeddingFunction(
        model=cfg.embedding_model
    )

    if not os.path.exists(here(cfg.vectordb_dir)):
        os.makedirs(here(cfg.vectordb_dir))
        print(f"Directory {cfg.vectordb_dir} was created.")

    db_client = chromadb.PersistentClient(path=cfg.vectordb_dir)
    db_collection = db_client.get_or_create_collection(
        name=cfg.collection_name,
        embedding_function=mistral_embedding_function,
        metadata={"hnsw:space": "cosine"}
    )
    print("DB Collection get created: ",db_collection)
    print("DB Collection count: ",db_collection.count())

if __name__ == "__main__":
    prepare_vectordb()
