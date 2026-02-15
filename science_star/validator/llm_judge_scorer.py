#!/usr/bin/env python

"""
LLM-as-a-Judge scorer: uses smolagents OpenAIServerModel.
"""

import os
from typing import Any, Dict, Optional

import json_repair
from smolagents import OpenAIServerModel

from validator.base import BaseScorer, EvaluationResult

JUDGE_PROMPT = """Judge whether the [response] correctly answers [question] based on [correct_answer].

[question]: {question}
[correct_answer]: {correct_answer}
[response]: {response}

Reply with a JSON object only: {{"correct": "yes" or "no", "reason": "<brief explanation>"}}"""


class LLMJudgeScorer(BaseScorer):
    """LLM-as-a-Judge using OpenAIServerModel."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = (config or {}).get("validator", config) or {}
        self.model = OpenAIServerModel(
            cfg.get("model", "gpt-4o-2024-11-20"),
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_BASE_URL"),
            temperature=cfg.get("temperature", 0.0),
            max_completion_tokens=cfg.get("max_tokens", 512),
        )

    def evaluate(
        self,
        ground_truth: str,
        model_response: str,
        question: Optional[str] = None,
    ) -> EvaluationResult:
        question = question or "(No question provided)"
        prompt = JUDGE_PROMPT.format(
            question=question,
            correct_answer=ground_truth,
            response=model_response,
        )
        msg = self.model.generate([{"role": "user", "content": prompt}])
        content = msg.content or ""
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        content = str(content) if not isinstance(content, str) else content

        obj = json_repair.loads(content) if content else {}
        obj = obj if isinstance(obj, dict) else {}
        is_correct = (obj.get("correct") or "").lower() == "yes"
        reason = obj.get("reason") or ""

        return EvaluationResult(is_correct=is_correct, reason=reason)
