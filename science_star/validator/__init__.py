"""Validator: unified entry point for answer evaluation."""

from typing import Any, Dict, Optional

from validator.base import BaseScorer, EvaluationResult
from validator.gaia_scorer import GaiaScorer
from validator.hle_scorer import HLEScorer
from validator.llm_judge_scorer import LLMJudgeScorer


def get_scorer(
    dataset: str,
    config: Optional[Dict[str, Any]] = None,
) -> BaseScorer:
    """
    Get the appropriate scorer for a dataset.

    Args:
        dataset: "gaia" | "hle" | "llm_judge"
        config: Optional config dict. For llm_judge/hle: model, temperature, max_tokens.

    Returns:
        BaseScorer instance.

    Examples:
        scorer = get_scorer("gaia")
        result = scorer.evaluate(ground_truth, model_response)

        scorer = get_scorer("hle", config={"model": "gpt-4o"})
        result = scorer.evaluate(ground_truth, model_response, question=question)
    """
    config = config or {}

    if dataset == "gaia":
        return GaiaScorer(config)
    if dataset == "hle":
        return HLEScorer(config)
    else:
        return LLMJudgeScorer(config)


__all__ = [
    "BaseScorer",
    "EvaluationResult",
    "GaiaScorer",
    "HLEScorer",
    "LLMJudgeScorer",
    "get_scorer",
]
