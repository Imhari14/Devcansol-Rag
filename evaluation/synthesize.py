"""
Synthetic QA pair generation module.
Uses DeepEval's Synthesizer to generate test data from domain documents.
"""

import logging
from pathlib import Path

from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig
from deepeval.models import GeminiModel
from deepeval.dataset import EvaluationDataset

from app.config import get_api_key
from app.gemini_embedder import GeminiEmbeddingModel

logger = logging.getLogger(__name__)


def generate_goldens_from_documents(
    document_paths: list[str],
    output_alias: str = "RAG QA Agent Dataset",
) -> list:
    """Generate synthetic question-answer pairs from documents.

    Uses DeepEval's Synthesizer to create goldens (input + expected_output)
    from the provided document files.

    Args:
        document_paths: List of file paths to source documents.
        output_alias: Name for the evaluation dataset.

    Returns:
        List of Golden objects with input and expected_output fields.
    """
    logger.info(
        "[synthesize] starting golden generation - documents=%d",
        len(document_paths),
    )

    for path in document_paths:
        if not Path(path).exists():
            logger.error("[synthesize] document not found - path=%s", path)
            raise FileNotFoundError(f"Document not found: {path}")

    synthesizer = Synthesizer(
        model=GeminiModel(
            model="gemini-2.5-flash",
            api_key=get_api_key(),
            temperature=0,
        )
    )
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=document_paths,
        context_construction_config=ContextConstructionConfig(
            embedder=GeminiEmbeddingModel(),
        ),
    )

    logger.info(
        "[synthesize] goldens generated - count=%d",
        len(goldens),
    )

    dataset = EvaluationDataset(goldens=goldens)

    logger.info(
        "[synthesize] dataset created - alias=%s, goldens=%d",
        output_alias,
        len(goldens),
    )
    return goldens, dataset
