"""Configuration and initialization helpers for recog-ai-demo."""

import os
import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv


def load_env():
    """Load environment variables from .env file."""
    load_dotenv()


def get_embedding():
    """Initialize and return the embedding model."""
    return HuggingFaceEmbeddings(
        model_name="isy-thl/multilingual-e5-base-course-skill-tuned",
        encode_kwargs={"normalize_embeddings": True, "prompt": "passage: "},
    )


def get_module_database(embedding, vectorstore_path: str = None):
    """Initialize and return the Chroma vector database for modules."""
    if vectorstore_path is None:
        vectorstore_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "modules_vectorstore"
        )

    return Chroma(
        client=chromadb.PersistentClient(vectorstore_path),
        embedding_function=embedding,
        client_settings=Settings(anonymized_telemetry=False),
    )
