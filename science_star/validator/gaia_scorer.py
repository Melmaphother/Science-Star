#!/usr/bin/env python

"""
GAIA dataset scorer: rule-based evaluation (no LLM).
"""

import re
import string
import warnings
from typing import Any, Dict, Optional

from validator.base import BaseScorer, EvaluationResult


class GaiaScorer(BaseScorer):
    """GAIA dataset scorer: rule-based, no LLM."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        pass

    def _normalize_number_str(self, number_str: str) -> float:
        for char in ["$", "%", ","]:
            number_str = number_str.replace(char, "")
        try:
            return float(number_str)
        except ValueError:
            return float("inf")

    def _split_string(self, s: str, char_list: Optional[list] = None) -> list[str]:
        char_list = char_list or [",", ";"]
        pattern = f"[{''.join(char_list)}]"
        return re.split(pattern, s)

    def _is_float(self, element) -> bool:
        try:
            float(element)
            return True
        except (ValueError, TypeError):
            return False

    def _normalize_str(self, input_str: str, remove_punct: bool = True) -> str:
        no_spaces = re.sub(r"\s", "", input_str)
        if remove_punct:
            translator = str.maketrans("", "", string.punctuation)
            return no_spaces.lower().translate(translator)
        return no_spaces.lower()

    def _score(self, model_answer: str, ground_truth: str) -> bool:
        if self._is_float(ground_truth):
            normalized = self._normalize_number_str(str(model_answer))
            return normalized == float(ground_truth)

        if any(char in ground_truth for char in [",", ";"]):
            gt_elems = self._split_string(ground_truth)
            ma_elems = self._split_string(model_answer)
            if len(gt_elems) != len(ma_elems):
                warnings.warn("Answer lists have different lengths.", UserWarning)
                return False
            comparisons = []
            for ma_elem, gt_elem in zip(ma_elems, gt_elems):
                if self._is_float(gt_elem):
                    comparisons.append(
                        self._normalize_number_str(ma_elem) == float(gt_elem)
                    )
                else:
                    comparisons.append(
                        self._normalize_str(ma_elem, remove_punct=False)
                        == self._normalize_str(gt_elem, remove_punct=False)
                    )
            return all(comparisons)

        return self._normalize_str(model_answer) == self._normalize_str(ground_truth)

    def _letters_in_order(self, prediction: str, true_answer: str) -> bool:
        prediction = prediction.lower()
        true_answer = true_answer.lower()
        if len(prediction) > len(true_answer) * 3:
            return False
        i = 0
        for letter in true_answer:
            if letter in prediction[i:]:
                i += prediction[i:].index(letter)
            else:
                return False
        return True

    def _check_close_call(
        self, prediction: str, true_answer: str, is_correct: bool
    ) -> bool:
        if is_correct:
            return True
        if self._is_float(true_answer):
            return False
        pred_str, gt_str = str(prediction), str(true_answer)
        return (
            self._letters_in_order(pred_str, gt_str)
            and len(gt_str) * 0.5 <= len(pred_str) <= len(gt_str) * 2
        )

    def evaluate(
        self,
        ground_truth: str,
        model_response: str,
        question: Optional[str] = None,
    ) -> EvaluationResult:
        is_correct = self._score(model_response, ground_truth)
        is_correct = self._check_close_call(model_response, ground_truth, is_correct)
        return EvaluationResult(
            is_correct=is_correct,
            reason="Rule-based evaluation (GAIA format).",
        )
