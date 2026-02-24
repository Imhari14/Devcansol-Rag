"""
Custom Gemini embedding model for DeepEval.
Wraps Google's Gemini Embedding API to work with DeepEval's Synthesizer.
"""

import logging
from typing import List

from google import genai
from google.genai import types
from deepeval.models import DeepEvalBaseEmbeddingModel

from app.config import get_api_key

logger = logging.getLogger(__name__)


class GeminiEmbeddingModel(DeepEvalBaseEmbeddingModel):
    """Custom embedding model using Google Gemini Embedding API.

    Implements DeepEval's base embedding interface to allow the Synthesizer
    to use Gemini embeddings instead of OpenAI.
    """

    def __init__(self):
        """Initialize the Gemini embedding client."""
        self.client = genai.Client(api_key=get_api_key())
        self.model_name = "gemini-embedding-001"

    def load_model(self):
        """Return the genai client as the model instance.

        Returns:
            The Google GenAI client.
        """
        return self.client

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768,
            ),
        )
        return result.embeddings[0].values

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple text strings.

        Args:
            texts: List of texts to embed.

        Returns:
            A list of embedding vectors.
        """
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768,
            ),
        )
        return [e.values for e in result.embeddings]

    async def a_embed_text(self, text: str) -> List[float]:
        """Async version of embed_text.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        return self.embed_text(text)

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed_texts.

        Args:
            texts: List of texts to embed.

        Returns:
            A list of embedding vectors.
        """
        return self.embed_texts(texts)

    def get_model_name(self) -> str:
        """Return the model name.

        Returns:
            The Gemini embedding model identifier.
        """
        return self.model_name
