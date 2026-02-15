"""GAIA dataset loader."""

import pandas as pd
from loguru import logger

from data.utils import parse_selected_tasks


def load_gaia_dataset(config) -> pd.DataFrame:
    """
    Load GAIA dataset based on config.dataset parameters.

    Expects config with:
        - config.repo_root: Path to repo root
        - config.dataset.subset: null | small | medium | large
        - config.dataset.level: null | level1 | level2 | level3
        - config.dataset.selected_tasks: optional list of task IDs or 1-based
            indices (1 = first question)

    Returns:
        pandas.DataFrame: Loaded and processed dataset.

    Raises:
        ValueError: If both subset and level are null, or both are not null.
        FileNotFoundError: If the specified data file does not exist.
    """
    ds = getattr(config, "dataset", config)
    subset = getattr(ds, "subset", None)
    level = getattr(ds, "level", None)
    selected_tasks = parse_selected_tasks(getattr(ds, "selected_tasks", None) or [])

    data_dir = config.repo_root / "data" / "GAIA"

    # When selected_tasks is specified, load from original file (not subset/level)
    if selected_tasks:
        data_path = data_dir / "gaia.jsonl"
        logger.info("Loading selected tasks from original file: {}", data_path)
    elif (subset is None) == (level is None):
        raise ValueError(
            "Exactly one of 'subset' or 'level' must be null, not both or neither."
        )
    elif subset is not None:
        subset_mapping = {
            "small": "gaia_subset_20.jsonl",
            "medium": "gaia_subset_50.jsonl",
            "large": "gaia_subset_100.jsonl",
        }
        if subset not in subset_mapping:
            raise ValueError(
                f"Invalid subset '{subset}'. "
                f"Must be one of: {list(subset_mapping.keys())}"
            )
        data_path = data_dir / "subset" / subset_mapping[subset]
        logger.info("Loading subset: {} from {}", subset, data_path)
    else:
        level_mapping = {
            "level1": "Level1.jsonl",
            "level2": "Level2.jsonl",
            "level3": "Level3.jsonl",
        }
        if level not in level_mapping:
            raise ValueError(
                f"Invalid level '{level}'. "
                f"Must be one of: {list(level_mapping.keys())}"
            )
        data_path = data_dir / "level" / level_mapping[level]
        logger.info("Loading level: {} from {}", level, data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info("Reading: {}", data_path)
    eval_df = pd.read_json(data_path, lines=True)

    column_mapping = {
        "task_id": "id",
        "Question": "question",
        "Final answer": "answer",
        "category": "category",
    }
    eval_df = eval_df.rename(columns=column_mapping)
    eval_df["true_answer"] = eval_df["answer"]  # alias for run script compatibility

    # Filter by selected_tasks if specified (indices refer to original file)
    # Integer indices are 1-based (1 = first question) for human readability
    if selected_tasks:
        if isinstance(selected_tasks[0], int):
            indices = [i - 1 for i in selected_tasks]
            if any(i < 0 for i in indices):
                raise ValueError(
                    "selected_tasks indices must be >= 1 (1-based). "
                    "Use 1 for the first question."
                )
            eval_df = eval_df.iloc[indices]
        else:
            eval_df = eval_df[eval_df["id"].isin(selected_tasks)]

    logger.info("Loaded {} samples", len(eval_df))
    if "category" in eval_df.columns:
        logger.info("Level distribution:\n{}", eval_df["category"].value_counts())

    return eval_df
