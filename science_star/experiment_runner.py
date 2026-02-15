#!/usr/bin/env python

"""Base experiment runner.

Provides config loading, experiment directory management, dataset loading,
task execution, and result recording. Subclasses implement create_agent()
to configure tools and agent framework.
"""

from __future__ import annotations

import json
import os
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf, DictConfig
from smolagents import OpenAIServerModel
from smolagents.memory import ActionStep, PlanningStep, TaskStep
from tqdm import tqdm

from data.gaia_loader import load_gaia_dataset
from data.hle_loader import load_hle_dataset
from io_processor.file_context import get_single_file_description, get_zip_description
from io_processor.reformulator import prepare_response
from templates import (
    AUGMENTED_QUESTION_PREFIX,
    FILE_ATTACHMENT_MULTIPLE_PREFIX,
    FILE_ATTACHMENT_SINGLE_PREFIX,
)
from tools.inspector.audio_inspector_tool import AudioInspectorTool
from tools.inspector.document_inspector_tool import DocumentInspectorTool
from tools.inspector.visual_inspector_tool import VisualInspectorTool
from loguru import logger
from validator import get_scorer

_result_lock = threading.Lock()

# Dataset name -> loader(config) -> pd.DataFrame
DATASET_LOADERS = {
    "gaia": load_gaia_dataset,
    "hle": load_hle_dataset,
}


class BaseAgentRunner(ABC):
    """Base class for benchmark experiment runners.

    Handles config loading, experiment directory setup, dataset loading,
    task execution, and result recording. Subclasses implement create_agent()
    to define the agent architecture and tools.
    """

    def __init__(self, repo_root: Path) -> None:
        """Initialize runner with project root path."""
        self.repo_root = Path(repo_root)
        self.config = self.load_config()

    def load_config(self, default_config_name: str = "gaia.yaml") -> DictConfig:
        """Load YAML config and merge with CLI overrides.

        Args:
            default_config_name: Default config file name under configs/.

        Returns:
            Merged OmegaConf DictConfig.
        """
        default_path = self.repo_root / "configs" / default_config_name
        cli_overrides = OmegaConf.from_cli()
        config_path = Path(str(cli_overrides.get("config", default_path)))
        base_config = OmegaConf.load(str(config_path))
        config = OmegaConf.merge(base_config, cli_overrides)
        selected_tasks = OmegaConf.select(config, "dataset.selected_tasks")
        if selected_tasks in (None, "null", "None"):
            OmegaConf.update(config, {"dataset": {"selected_tasks": []}}, merge=True)
        return config

    def create_model(self) -> OpenAIServerModel:
        """Build OpenAIServerModel from self.config.models."""
        models_config = getattr(self.config, "models", self.config)
        return OpenAIServerModel(
            model_id=getattr(models_config, "name", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_BASE_URL"),
            custom_role_conversions={
                "tool-call": "assistant",
                "tool-response": "user",
            },
            max_completion_tokens=getattr(models_config, "max_tokens", 4096),
            temperature=getattr(models_config, "temperature", 0.8),
        )

    @abstractmethod
    def create_agent(self, model: OpenAIServerModel) -> Any:
        """Create the agent with configured tools and framework.

        Args:
            model: LLM model for the agent.

        Returns:
            Agent instance (ToolCallingAgent, CodeAgent, etc.).
        """
        ...

    def _extract_intermediate_steps(self, agent: Any) -> list[dict]:
        """Extract intermediate steps from agent memory for JSONL serialization."""
        steps = []
        for memory_step in agent.memory.steps:
            step_dict = memory_step.dict()
            step_dict.pop("model_input_messages", None)
            if isinstance(memory_step, ActionStep):
                step_dict["step_type"] = "action"
                step_dict.pop("model_output_message", None)
            elif isinstance(memory_step, TaskStep):
                step_dict["step_type"] = "task"
            elif isinstance(memory_step, PlanningStep):
                step_dict["step_type"] = "planning"
                step_dict.pop("model_output_message", None)
            else:
                step_dict["step_type"] = "unknown"
            steps.append(step_dict)
        return steps

    def _extra_result_fields(self, agent: Any, example: dict) -> dict[str, Any]:
        """Return extra fields to add to the result entry. Override in subclasses."""
        return {}

    def _append_result(self, entry: dict, answers_path: Path) -> None:
        """Append result to JSONL file."""
        answers_path.parent.mkdir(parents=True, exist_ok=True)
        with _result_lock:
            with open(answers_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    def _build_augmented_question(
        self, example: dict, model: OpenAIServerModel
    ) -> str:
        """Build question string with optional file attachment context."""
        question = AUGMENTED_QUESTION_PREFIX + example["question"]
        if not example.get("file_name"):
            return question

        inspector_tools = (
            VisualInspectorTool(model),
            DocumentInspectorTool(model),
            AudioInspectorTool(model),
        )
        if ".zip" in example["file_name"]:
            prefix = FILE_ATTACHMENT_MULTIPLE_PREFIX
            prefix += get_zip_description(
                example["file_name"], example["question"], *inspector_tools
            )
        else:
            prefix = FILE_ATTACHMENT_SINGLE_PREFIX
            prefix += get_single_file_description(
                example["file_name"], example["question"], *inspector_tools
            )
        return question + prefix

    def _get_pending_tasks(
        self,
        answers_path: Path,
        eval_df: pd.DataFrame,
        debug: bool = False,
    ) -> list[dict]:
        """Get tasks not yet completed (excluding those in answers file)."""
        logger.info("Loading answers from {}...", answers_path)
        done_ids = []
        try:
            if answers_path.exists() and answers_path.stat().st_size > 0:
                answer_df = pd.read_json(answers_path, lines=True)
                done_ids = answer_df.get("id", []).tolist()
                logger.info("Found {} previous results", len(done_ids))
            else:
                logger.info("No prior results; starting fresh.")
        except Exception as e:
            logger.warning("Error loading answers: {}. Starting fresh.", e)

        if debug:
            done_ids = []

        return [
            row.to_dict()
            for _, row in eval_df.iterrows()
            if row["id"] not in done_ids
        ]

    def _setup_experiment_dirs(self) -> tuple[Path, Path, Path]:
        """Create run directories and return (run_time_dir, answers_path, config_path)."""
        self.config.repo_root = self.repo_root
        output_dir = self.repo_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        runtime_config = getattr(self.config, "runtime", self.config)
        run_name = getattr(runtime_config, "run_name", "run")
        if " " in run_name or "/" in run_name:
            run_name = run_name.replace(" ", "_").replace("/", "_")
            logger.warning("Sanitized run_name to {}", run_name)
            OmegaConf.update(self.config, {"runtime": {"run_name": run_name}}, merge=True)

        run_dir = output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        date_time_load = getattr(runtime_config, "date_time_load_from", None)
        if date_time_load:
            if not (run_dir / date_time_load).exists():
                raise FileNotFoundError(
                    f"Experiment dir not found: {run_dir / date_time_load}"
                )
            run_time_dir = run_dir / date_time_load
        else:
            run_time_dir = run_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
            run_time_dir.mkdir(parents=True, exist_ok=True)

        answers_path = run_time_dir / "answers.jsonl"
        config_path = run_time_dir / "config.yaml"
        return run_time_dir, answers_path, config_path

    def _load_dataset(self) -> pd.DataFrame:
        """Load dataset by self.config.dataset.name."""
        dataset_name = getattr(
            getattr(self.config, "dataset", None), "name", "gaia"
        )
        loader = DATASET_LOADERS.get(dataset_name)
        if loader is None:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Available: {list(DATASET_LOADERS.keys())}."
            )
        return loader(self.config)

    def run_task(
        self,
        example: dict,
        answers_path: Path,
        debug: bool = False,
    ) -> None:
        """Run agent on one task and append result to answers file."""
        logger.info("Starting task {}: {}", example["id"], example["question"][:80] + ("..." if len(example["question"]) > 80 else ""))
        model = self.create_model()
        agent = self.create_agent(model)
        augmented_question = self._build_augmented_question(example, model)

        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            final_result = agent.run(augmented_question)
            agent_memory = agent.write_memory_to_messages(summary_mode=True)
            final_result = prepare_response(
                augmented_question, agent_memory, reformulation_model=model
            )
            output = str(final_result)

            intermediate_steps = self._extract_intermediate_steps(agent)
            step_strs = [str(s) for s in agent.memory.steps]
            parsing_error = any("AgentParsingError" in s for s in step_strs)
            iteration_limit_exceeded = (
                "Agent stopped due to iteration limit or time limit." in output
            )
            raised_exception = False

        except Exception as e:
            logger.error("Error on task {}: {}", example["id"], e)
            output = None
            intermediate_steps = []
            parsing_error = False
            iteration_limit_exceeded = False
            exception = e
            raised_exception = True

        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        judgment_result = None
        dataset_name = getattr(
            getattr(self.config, "dataset", None), "name", "gaia"
        )
        if output and not raised_exception:
            try:
                scorer = get_scorer(
                    dataset_name,
                    config=OmegaConf.to_container(self.config, resolve=True),
                )
                result = scorer.evaluate(
                    ground_truth=example["true_answer"],
                    model_response=output,
                )
                judgment_result = result.to_dict()
            except Exception as e:
                logger.warning("Scoring failed for task {}: {}", example["id"], e)

        entry = {
            "agent_name": getattr(getattr(self.config, "models", None), "name", "unknown"),
            "question": example["question"],
            "augmented_question": augmented_question,
            "prediction": output,
            "true_answer": example["answer"],
            "intermediate_steps": intermediate_steps,
            "parsing_error": parsing_error,
            "iteration_limit_exceeded": iteration_limit_exceeded,
            "agent_error": str(exception) if raised_exception else None,
            "start_time": start_time,
            "end_time": end_time,
            "id": example["id"],
            "judgment_result": judgment_result,
        }
        if "category" in example:
            entry["category"] = example["category"]
        entry.update(self._extra_result_fields(agent, example))
        self._append_result(entry, answers_path)
        logger.info("Task {} completed, prediction: {}", example["id"], (output[:100] + "...") if output and len(str(output)) > 100 else output)

    def run(self) -> None:
        """Main entry: setup dirs, load dataset, run tasks."""
        logger.info("Config: {}", OmegaConf.to_container(self.config, resolve=True))

        run_time_dir, answers_path, config_path = self._setup_experiment_dirs()
        with open(config_path, "w") as f:
            OmegaConf.save(config=self.config, f=f)

        eval_df = self._load_dataset()
        runtime_config = getattr(self.config, "runtime", self.config)
        debug = getattr(runtime_config, "debug", False)
        tasks = self._get_pending_tasks(answers_path, eval_df, debug)
        concurrency = getattr(runtime_config, "concurrency", 1)

        logger.info("Output dir: {}", run_time_dir)
        logger.info("Answers file: {}", answers_path)
        logger.info("Running {} tasks", len(tasks))

        if debug or concurrency == 1:
            for example in tasks:
                self.run_task(example, answers_path, debug)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(
                        self.run_task, example, answers_path, debug
                    )
                    for example in tasks
                ]
                for future in tqdm(
                    as_completed(futures),
                    total=len(tasks),
                    desc="Processing tasks",
                ):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error("Task failed: {}", e)

        logger.info("All tasks processed.")
