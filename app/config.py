"""
Application configuration module.
Provides centralized config access with runtime API key support.
"""

import os
import logging

logger = logging.getLogger(__name__)


# --- Mutable API Key (set at runtime from UI) ---
GOOGLE_API_KEY = ""

# --- Model Configuration ---
GENERATION_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768

# --- RAG Configuration ---
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 3

# --- Paths ---
VECTOR_STORE_DIR = "vector_store"
DOCUMENTS_DIR = "documents"
EVALUATION_RESULTS_DIR = "evaluation_results"


def set_api_key(api_key: str) -> None:
    """Set the Google API key at runtime.

    Updates the module-level variable and os.environ so all
    third-party libraries can discover the key.

    Args:
        api_key: The Google API key string.
    """
    global GOOGLE_API_KEY
    GOOGLE_API_KEY = api_key
    os.environ["GOOGLE_API_KEY"] = api_key
    logger.info("[config] API key set at runtime")


def get_api_key() -> str:
    """Return the current Google API key.

    Returns:
        The API key string, or empty string if not set.
    """
    return GOOGLE_API_KEY


def validate_config() -> bool:
    """Validate that all required configuration values are present.

    Returns:
        True if configuration is valid, False otherwise.
    """
    if not GOOGLE_API_KEY:
        logger.error("[config] GOOGLE_API_KEY is not set")
        return False

    logger.info("[config] configuration validated successfully")
    return True
