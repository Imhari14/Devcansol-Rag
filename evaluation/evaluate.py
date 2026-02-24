"""
RAG evaluation module.
Runs retriever and generator metrics using DeepEval against synthetic test data.
"""

import logging
import json
from datetime import datetime
from pathlib import Path

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    GEval,
)

from app.rag_agent import RAGAgent
from app.config import EVALUATION_RESULTS_DIR, get_api_key

logger = logging.getLogger(__name__)


def _get_gemini_judge():
    """Create a GeminiModel instance for use as evaluation judge.

    Returns:
        GeminiModel instance configured with the project API key.
    """
    from deepeval.models import GeminiModel
    return GeminiModel(
        model="gemini-2.5-flash",
        api_key=get_api_key(),
        temperature=0,
    )


def build_retriever_metrics(threshold: float = 0.3) -> list:
    """Create the three retriever evaluation metrics.

    Args:
        threshold: Minimum score to pass. Defaults to 0.5.

    Returns:
        List of retriever metric instances.
    """
    judge = _get_gemini_judge()
    return [
        ContextualRecallMetric(threshold=threshold, model=judge),
        ContextualPrecisionMetric(threshold=threshold, model=judge),
    ]


def build_generator_metrics(threshold: float = 0.3) -> list:
    """Create the generator evaluation metrics.

    Args:
        threshold: Minimum score to pass. Defaults to 0.5.

    Returns:
        List of generator metric instances.
    """
    judge = _get_gemini_judge()

    answer_correctness = GEval(
        name="Answer Correctness",
        model=judge,
        criteria=(
            "Evaluate if the actual output's answer is correct and complete "
            "based on the input and retrieved context. If the answer is not "
            "correct or complete, reduce the score."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        threshold=threshold,
    )

    return [
        FaithfulnessMetric(threshold=threshold, model=judge),
        AnswerRelevancyMetric(threshold=threshold, model=judge),
        answer_correctness,
    ]


def create_test_cases(
    agent: RAGAgent,
    goldens: list,
) -> list[LLMTestCase]:
    """Create LLMTestCase objects by running the RAG agent on golden inputs.

    Args:
        agent: The RAGAgent instance to evaluate.
        goldens: List of Golden objects with input and expected_output.

    Returns:
        List of LLMTestCase objects ready for evaluation.
    """
    test_cases = []

    for i, golden in enumerate(goldens):
        retrieved_docs = agent.retrieve(golden.input)
        result = agent.answer(golden.input)
        actual_output = json.dumps(result) if isinstance(result, dict) else str(result)

        test_case = LLMTestCase(
            input=golden.input,
            actual_output=actual_output,
            retrieval_context=retrieved_docs,
            expected_output=golden.expected_output,
        )
        test_cases.append(test_case)

        logger.info(
            "[evaluate] test case created - index=%d, input_length=%d",
            i,
            len(golden.input),
        )

    logger.info("[evaluate] all test cases created - total=%d", len(test_cases))
    return test_cases


def run_evaluation(
    test_cases: list[LLMTestCase],
    evaluation_type: str = "all",
    threshold: float = 0.3,
) -> dict:
    """Run DeepEval metrics on the provided test cases.

    Args:
        test_cases: List of LLMTestCase objects.
        evaluation_type: One of 'retriever', 'generator', or 'all'.
        threshold: Minimum score threshold for metrics.

    Returns:
        Dictionary containing evaluation results and metadata.
    """
    metrics = []

    if evaluation_type in ("retriever", "all"):
        metrics.extend(build_retriever_metrics(threshold))
    if evaluation_type in ("generator", "all"):
        metrics.extend(build_generator_metrics(threshold))

    logger.info(
        "[evaluate] starting evaluation - type=%s, test_cases=%d, metrics=%d",
        evaluation_type,
        len(test_cases),
        len(metrics),
    )

    results = evaluate(test_cases, metrics)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_summary = {
        "timestamp": timestamp,
        "evaluation_type": evaluation_type,
        "test_cases_count": len(test_cases),
        "threshold": threshold,
    }

    logger.info(
        "[evaluate] evaluation completed - type=%s, timestamp=%s",
        evaluation_type,
        timestamp,
    )

    output_dir = Path(EVALUATION_RESULTS_DIR)
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"eval_{evaluation_type}_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    logger.info("[evaluate] results saved - path=%s", output_path)
    return results_summary
