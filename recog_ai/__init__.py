"""Package initialization for recog_ai module."""

from recog_ai.config import load_env, get_embedding, get_module_database
from recog_ai.llm_client import LLMClient
from recog_ai.assistant import RecognitionAssistant
from recog_ai.utils import extract_json, parse_workload, collect_programs

# Load environment at module import
load_env()

# For backward compatibility, expose the old class name
recognition_assistant = RecognitionAssistant

__all__ = [
    "load_env",
    "get_embedding",
    "get_module_database",
    "LLMClient",
    "RecognitionAssistant",
    "recognition_assistant",
    "extract_json",
    "parse_workload",
    "collect_programs",
]
