"""Shared utilities for dataset loaders."""

import json
import os


def parse_selected_tasks(tasks_param) -> list:
    """Parse dataset.selected_tasks from config into a list of task IDs/indices.

    Supports: (1) file path (.jsonl or .txt) - reads IDs from file;
    (2) single int/str - returns [item]; (3) list of ids - returns as-is.
    Empty list means run all tasks.

    Args:
        tasks_param: Raw config value - list, file path, or single item.

    Returns:
        List of task IDs (str) or indices (int).
    """
    if not tasks_param:
        return []
    if len(tasks_param) == 1:
        item = tasks_param[0]
        if os.path.isfile(item):
            if item.endswith(".jsonl"):
                with open(item, "r", encoding="utf-8") as f:
                    return [json.loads(line)["id"] for line in f if "id" in json.loads(line)]
            with open(item, "r") as f:
                return [line.strip() for line in f if line.strip()]
        try:
            return [int(item)]
        except ValueError:
            return [item]
    result = []
    for item in tasks_param:
        try:
            result.append(int(item))
        except ValueError:
            result.append(item)
    return result
