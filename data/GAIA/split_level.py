#!/usr/bin/env python3
"""
Split GAIA dataset by Level into level/ folder.

Reads from gaia_validation.jsonl (or gaia.jsonl) and writes:
- level/Level1.jsonl
- level/Level2.jsonl
- level/Level3.jsonl

Usage:
    python data/GAIA/split_level.py
    python data/GAIA/split_level.py --input data/GAIA/gaia_test.jsonl
"""

import argparse
import json
from pathlib import Path


def split_by_level(
    input_path: Path,
    output_dir: Path,
) -> None:
    """
    Split GAIA JSONL by Level column.

    Args:
        input_path: Path to gaia_validation.jsonl or gaia.jsonl.
        output_dir: Output directory for level/*.jsonl files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    level_files: dict[str, list] = {"1": [], "2": [], "3": []}

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            level = str(rec.get("Level", "")).strip()
            if level in level_files:
                level_files[level].append(rec)

    for level, records in level_files.items():
        out_path = output_dir / f"Level{level}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"✅ Level{level}: {len(records)} samples -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split GAIA by Level")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input JSONL (default: gaia_validation.jsonl or gaia.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: level/)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir or (script_dir / "level")
    input_path = args.input
    if input_path is None:
        for name in ["gaia_validation.jsonl", "gaia.jsonl"]:
            candidate = script_dir / name
            if candidate.exists():
                input_path = candidate
                break
        if input_path is None:
            raise FileNotFoundError(
                "No GAIA file found. Ensure gaia.jsonl or gaia_validation.jsonl exists in data/GAIA/"
            )

    split_by_level(input_path, output_dir)


if __name__ == "__main__":
    main()
