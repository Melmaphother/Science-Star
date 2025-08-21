#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The OPPO Inc. Personal AI team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Answer Scoring and Evaluation Module

This module provides comprehensive answer evaluation capabilities for AI agents,
supporting both rule-based and LLM-based judging methods. It includes:
- Rule-based scoring for exact matches, numerical values, and lists
- LLM judge functionality using OpenAI models for nuanced evaluation
- Calibration metrics and accuracy calculations
- Configurable evaluation parameters and custom prompts
"""

import json
import re
import string
import warnings
import asyncio
import math
from typing import Literal, Optional, Dict, Any, Union, List

import numpy as np
from pydantic import BaseModel

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None
    print("Warning: OpenAI not installed. LLM judge functionality will be disabled.")

try:
    import os
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Environment variables may not be loaded.")

# ============================================================================
# LLM Judge Configuration
# ============================================================================
# Configuration classes and utilities for LLM-based answer evaluation

class LLMJudgeConfig:
    """Configuration class for LLM judge functionality."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "o3-mini-2025-01-31",
        timeout: float = 300.0,
        max_retries: int = 1,
        max_completion_tokens: int = 4096,
        temperature: float = 0.0,
        organization: Optional[str] = None,
        project: Optional[str] = None
    ):
        """Initialize LLM judge configuration.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Custom API base URL (defaults to OPENAI_BASE_URL env var)
            model: Model to use for judging
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            max_completion_tokens: Maximum tokens for completion
            temperature: Sampling temperature (0.0 for deterministic)
            organization: OpenAI organization ID
            project: OpenAI project ID
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.organization = organization or os.getenv("OPENAI_ORG_ID")
        self.project = project or os.getenv("OPENAI_PROJECT_ID")
        
    def create_client(self) -> Optional["AsyncOpenAI"]:
        """Create AsyncOpenAI client with current configuration."""
        if AsyncOpenAI is None:
            print("Error: OpenAI not available. Cannot create client.")
            return None
            
        client_kwargs = {
            "timeout": self.timeout,
            "max_retries": self.max_retries
        }
        
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        if self.organization:
            client_kwargs["organization"] = self.organization
        if self.project:
            client_kwargs["project"] = self.project
            
        try:
            return AsyncOpenAI(**client_kwargs)
        except Exception as e:
            print(f"Error creating OpenAI client: {e}")
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": self.model,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "max_completion_tokens": self.max_completion_tokens,
            "temperature": self.temperature,
            "base_url": self.base_url,
            "organization": self.organization,
            "project": self.project
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LLMJudgeConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_env(cls) -> "LLMJudgeConfig":
        """Create configuration from environment variables."""
        return cls(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            model=os.getenv("LLM_JUDGE_MODEL", "o3-mini-2025-01-31"),
            timeout=float(os.getenv("LLM_JUDGE_TIMEOUT", "300.0")),
            max_retries=int(os.getenv("LLM_JUDGE_MAX_RETRIES", "1")),
            max_completion_tokens=int(os.getenv("LLM_JUDGE_MAX_TOKENS", "4096")),
            temperature=float(os.getenv("LLM_JUDGE_TEMPERATURE", "0.0")),
            organization=os.getenv("OPENAI_ORG_ID"),
            project=os.getenv("OPENAI_PROJECT_ID")
        )

# Default configuration instance
default_llm_config = LLMJudgeConfig.from_env()

# ============================================================================
# Configuration Utility Functions
# ============================================================================
# Helper functions for creating and managing LLM judge configurations
def create_llm_config(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "o3-mini-2025-01-31",
    temperature: float = 0.0,
    max_completion_tokens: int = 4096,
    timeout: float = 300.0,
    max_retries: int = 1
) -> LLMJudgeConfig:
    """Create a custom LLM judge configuration.
    
    Args:
        api_key: OpenAI API key (uses env var if None)
        base_url: OpenAI base URL (uses env var if None)
        model: Model name to use for judging
        temperature: Sampling temperature (0.0 for deterministic)
        max_completion_tokens: Maximum tokens in response
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries
        
    Returns:
        Configured LLMJudgeConfig instance
    """
    return LLMJudgeConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        timeout=timeout,
        max_retries=max_retries
    )

def get_available_models() -> List[str]:
    """Get list of commonly used models for LLM judging.
    
    Returns:
        List of model names suitable for judging tasks
    """
    return [
        "o3-mini-2025-01-31",
        "gpt-4o-2024-11-20",
        "gpt-4o-mini-2024-07-18",
        "gpt-4-turbo-2024-04-09",
        "gpt-3.5-turbo-0125"
    ]

def validate_config(config: LLMJudgeConfig) -> bool:
    """Validate LLM judge configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        client = config.create_client()
        return client is not None
    except Exception:
        return False

# ============================================================================
# LLM Judge Prompt and Response Models
# ============================================================================
# Structured prompt template and Pydantic models for LLM-based evaluation

JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available."""

# Pydantic model for structured LLM judge responses
class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True] = True  # 100% reliability


# ============================================================================
# Rule-Based Scoring Functions
# ============================================================================
# Traditional scoring methods for exact matching and numerical comparison

def normalize_number_str(number_str: str) -> float:
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        print(f"String {number_str} cannot be normalized to number str.")
        return float("inf")


def split_string(
    s: str,
    char_list: list[str] = [",", ";"],
) -> list[str]:
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)


# Main rule-based scoring function supporting numbers, lists, and strings
def question_scorer(
    model_answer: str,
    ground_truth: str,
) -> bool:
    def is_float(element: any) -> bool:
        try:
            float(element)
            return True
        except ValueError:
            return False

    if is_float(ground_truth):
        print(f"Evaluating {model_answer} as a number.")
        normalized_answer = normalize_number_str(model_answer)
        return normalized_answer == float(ground_truth)

    elif any(char in ground_truth for char in [",", ";"]):
        print(f"Evaluating {model_answer} as a comma separated list.")

        gt_elems = split_string(ground_truth)
        ma_elems = split_string(model_answer)

        if len(gt_elems) != len(ma_elems):
            warnings.warn(
                "Answer lists have different lengths, returning False.", UserWarning
            )
            return False

        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if is_float(gt_elem):
                normalized_ma_elem = normalize_number_str(ma_elem)
                comparisons.append(normalized_ma_elem == float(gt_elem))
            else:
                comparisons.append(
                    normalize_str(ma_elem, remove_punct=False)
                    == normalize_str(gt_elem, remove_punct=False)
                )
        return all(comparisons)

    else:
        print(f"Evaluating {model_answer} as a string.")
        return normalize_str(model_answer) == normalize_str(ground_truth)


def normalize_str(input_str, remove_punct=True) -> str:
    """Normalize string by removing spaces and optionally punctuation."""
    no_spaces = re.sub(r"\s", "", input_str)

    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    else:
        return no_spaces.lower()


# ============================================================================
# LLM Judge Functions
# ============================================================================
# Advanced evaluation using language models for nuanced answer assessment

async def llm_judge_answer(
    question: str,
    model_answer: str,
    correct_answer: str,
    config: Optional[LLMJudgeConfig] = None,
    client: Optional["AsyncOpenAI"] = None,
    custom_prompt: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Use LLM to judge if model answer is correct.
    
    Args:
        question: The original question
        model_answer: The model's response to evaluate
        correct_answer: The ground truth answer
        config: LLM judge configuration (uses default if None)
        client: AsyncOpenAI client instance (will create from config if None)
        custom_prompt: Custom prompt template (uses default if None)
        
    Returns:
        Dictionary with judgment results or None if error
    """
    if AsyncOpenAI is None:
        print("Error: OpenAI not available. Cannot use LLM judge.")
        return None
    
    # Use default config if none provided
    if config is None:
        config = default_llm_config
        
    # Create client if none provided
    if client is None:
        client = config.create_client()
        if client is None:
            return None
    
    # Use custom prompt or default
    prompt_template = custom_prompt or JUDGE_PROMPT
    prompt = prompt_template.format(
        question=question,
        correct_answer=correct_answer,
        response=model_answer
    )
    
    try:
        response = await client.beta.chat.completions.parse(
            model=config.model,
            max_completion_tokens=config.max_completion_tokens,
            temperature=config.temperature,
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format=ExtractedAnswer,
        )
        content = response.choices[0].message.parsed
        return {
            "correct_answer": correct_answer,
            "model_answer": content.extracted_final_answer,
            "reasoning": content.reasoning,
            "correct": content.correct,
            "confidence": content.confidence,
            "is_correct": content.correct == "yes",
            "config_used": config.to_dict()
        }
    except Exception as e:
        print(f"Error in LLM judge: {e}")
        return None


def llm_judge_answer_sync(
    question: str,
    model_answer: str,
    correct_answer: str,
    config: Optional[LLMJudgeConfig] = None,
    client: Optional["AsyncOpenAI"] = None,
    custom_prompt: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Synchronous wrapper for LLM judge."""
    return asyncio.run(llm_judge_answer(
        question, model_answer, correct_answer, config, client, custom_prompt
    ))


# ============================================================================
# Unified Scoring Interface
# ============================================================================
# Main entry point supporting both rule-based and LLM-based evaluation methods

def score_answer(
    model_answer: str,
    ground_truth: str,
    question: Optional[str] = None,
    method: str = "rule_based",
    llm_config: Optional[LLMJudgeConfig] = None,
    llm_client: Optional["AsyncOpenAI"] = None,
    custom_prompt: Optional[str] = None,
    **kwargs
) -> Union[bool, Dict[str, Any]]:
    """Unified interface for scoring answers using different methods.
    
    Args:
        model_answer: The model's answer to evaluate
        ground_truth: The correct answer
        question: The original question (required for LLM judge)
        method: Scoring method - "rule_based" or "llm_judge"
        llm_config: LLM judge configuration (uses default if None)
        llm_client: AsyncOpenAI client for LLM judge
        custom_prompt: Custom prompt template for LLM judge
        **kwargs: Additional arguments for specific methods
        
    Returns:
        For rule_based: boolean indicating correctness
        For llm_judge: dictionary with detailed judgment results
    """
    if method == "rule_based":
        return question_scorer(model_answer, ground_truth)
    elif method == "llm_judge":
        if question is None:
            raise ValueError("Question is required for LLM judge method")
        return llm_judge_answer_sync(
            question, model_answer, ground_truth, llm_config, llm_client, custom_prompt
        )
    else:
        raise ValueError(f"Unknown scoring method: {method}")


# ============================================================================
# Calibration and Metrics Functions
# ============================================================================
# Statistical analysis tools for evaluating model confidence and accuracy

def calib_err(confidence: np.ndarray, correct: np.ndarray, p: str = '2', beta: int = 100) -> float:
    """Calculate calibration error.
    
    Args:
        confidence: Array of confidence scores (0-1)
        correct: Array of correctness indicators (boolean)
        p: Norm type ('1', '2', 'infty')
        beta: Target bin size
        
    Returns:
        Calibration error
    """
    if len(confidence) == 0:
        return 0.0
        
    # beta is target bin size
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    
    # Handle small datasets
    if len(confidence) < beta:
        # Use single bin for small datasets
        bins = [[0, len(confidence)]]
    else:
        bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
        if bins:  # Only modify if bins exist
            bins[-1] = [bins[-1][0], len(confidence)]
        else:
            bins = [[0, len(confidence)]]

    cerr = 0
    total_examples = len(confidence)
    
    for i in range(len(bins)):
        bin_confidence = confidence[bins[i][0]:bins[i][1]]
        bin_correct = correct[bins[i][0]:bins[i][1]]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))

            if p == '2':
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == '1':
                cerr += num_examples_in_bin / total_examples * difference
            elif p == 'infty' or p == 'infinity' or p == 'max':
                cerr = np.maximum(cerr, difference)
            else:
                assert False, "p must be '1', '2', or 'infty'"

    if p == '2':
        cerr = np.sqrt(cerr)

    return cerr


def calculate_metrics(predictions: Dict[str, Dict[str, Any]], total_questions: int) -> Dict[str, float]:
    """Calculate accuracy and calibration metrics from predictions.
    
    Args:
        predictions: Dictionary of predictions with judge responses
        total_questions: Total number of questions
        
    Returns:
        Dictionary with accuracy, confidence interval, and calibration error
    """
    correct = []
    confidence = []
    
    for k, v in predictions.items():
        if "judge_response" in v:
            judge_response = v["judge_response"]
            correct.append("yes" in judge_response["correct"])
            confidence.append(judge_response["confidence"])
        else:
            print(f"Missing judge response for {k}")

    correct = np.array(correct)
    confidence = np.array(confidence) / 100

    if len(correct) != total_questions:
        print(f"Available predictions: {len(correct)} | Total questions: {total_questions}")

    accuracy = 100 * sum(correct) / total_questions
    # Wald estimator, 95% confidence interval
    confidence_half_width = 1.96 * math.sqrt(accuracy * (100 - accuracy) / total_questions)
    calibration_error = 100 * calib_err(confidence, correct, p='2', beta=100)

    return {
        "accuracy": round(accuracy, 2),
        "confidence_interval": round(confidence_half_width, 2),
        "calibration_error": round(calibration_error, 2),
        "total_questions": total_questions,
        "available_predictions": len(correct)
    }


def print_metrics(metrics: Dict[str, float]) -> None:
    """Print formatted metrics."""
    print("*** Metrics ***")
    print(f"Accuracy: {metrics['accuracy']}% +/- {metrics['confidence_interval']}% | n = {metrics['total_questions']}")
    print(f"Calibration Error: {metrics['calibration_error']}")
    if metrics['available_predictions'] != metrics['total_questions']:
        print(f"Note: Only {metrics['available_predictions']} predictions available out of {metrics['total_questions']} total questions")


# ============================================================================
# Example Usage and Documentation
# ============================================================================
# Comprehensive examples demonstrating various scoring methods and configurations

"""
Example usage of the configurable LLM judge functionality:

# 1. Using default configuration (from environment variables)
result = score_answer(
    model_answer="The capital of France is Paris.",
    ground_truth="Paris",
    question="What is the capital of France?",
    method="llm_judge"
)

# 2. Creating custom configuration
custom_config = create_llm_config(
    model="gpt-4o-2024-11-20",
    temperature=0.1,
    max_completion_tokens=2048,
    timeout=120.0
)

result = score_answer(
    model_answer="The answer is 42.",
    ground_truth="42",
    question="What is the meaning of life?",
    method="llm_judge",
    llm_config=custom_config
)

# 3. Using custom prompt template
custom_prompt = '''Evaluate if the response correctly answers the question.
Question: {question}
Correct Answer: {correct_answer}
Response: {response}

Provide your judgment in the required format.'''

result = score_answer(
    model_answer="Machine learning is a subset of AI.",
    ground_truth="Machine learning is a branch of artificial intelligence.",
    question="What is machine learning?",
    method="llm_judge",
    custom_prompt=custom_prompt
)

# 4. Batch evaluation with metrics
responses = [
    {"question": "What is 2+2?", "model_answer": "4", "ground_truth": "4"},
    {"question": "What is the capital of Japan?", "model_answer": "Tokyo", "ground_truth": "Tokyo"},
    # ... more responses
]

judge_results = []
for resp in responses:
    result = score_answer(
        model_answer=resp["model_answer"],
        ground_truth=resp["ground_truth"],
        question=resp["question"],
        method="llm_judge"
    )
    if result:
        judge_results.append(result)

# Calculate and print metrics
metrics = calculate_metrics(judge_results)
print_metrics(metrics)

# 5. Environment variables for configuration
# Set these environment variables:
# OPENAI_API_KEY=your_api_key_here
# OPENAI_BASE_URL=https://api.openai.com/v1  # optional
# LLM_JUDGE_MODEL=gpt-4o-2024-11-20  # optional
# LLM_JUDGE_TEMPERATURE=0.0  # optional
# LLM_JUDGE_MAX_TOKENS=4096  # optional
# LLM_JUDGE_TIMEOUT=300.0  # optional
# LLM_JUDGE_MAX_RETRIES=1  # optional
"""
