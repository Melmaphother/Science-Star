#!/usr/bin/env python

"""
Base scorer interface for answer evaluation.

All dataset-specific scorers implement evaluate(ground_truth, model_response)
and return (is_correct, reason). The question is optional for LLM judges.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class EvaluationResult:
    """Unified output from any scorer."""

    is_correct: bool
    reason: str
    extracted_answer: Optional[str] = None
    confidence: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dict for JSON storage / compatibility."""
        d = {
            "is_correct": self.is_correct,
            "reason": self.reason,
            "reasoning": self.reason,  # alias for compatibility
            "correct": "yes" if self.is_correct else "no",
        }
        if self.extracted_answer is not None:
            d["extracted_answer"] = self.extracted_answer
            d["model_answer"] = self.extracted_answer
        if self.confidence is not None:
            d["confidence"] = self.confidence
        return d


class BaseScorer(ABC):
    """Abstract base class for answer evaluation."""

    @abstractmethod
    def evaluate(
        self,
        ground_truth: str,
        model_response: str,
        question: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate whether model_response matches ground_truth.

        Args:
            ground_truth: The correct answer.
            model_response: The model's answer to evaluate.
            question: Optional question context (used by LLM judges).

        Returns:
            EvaluationResult with is_correct and reason.
        """
        pass
