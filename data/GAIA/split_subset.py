#!/usr/bin/env python3
"""
Create proportional subsets of GAIA dataset (stratified by Level).

Reads from gaia_validation.jsonl (or gaia.jsonl) and writes:
- subset/gaia_subset_20.jsonl
- subset/gaia_subset_50.jsonl
- subset/gaia_subset_100.jsonl

GAIA (~165 samples) uses smaller subset sizes than HLE (50, 200, 500).

Usage:
    python data/GAIA/split_subset.py
    python data/GAIA/split_subset.py --input data/GAIA/gaia_validation.jsonl
"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


def _compute_proportional_allocation(
    level_to_count: Dict[str, int], sample_size: int
) -> Dict[str, int]:
    """
    Allocate samples per level using proportional distribution.
    Ensures at least 1 per level when possible.
    """
    total = sum(level_to_count.values())
    levels = list(level_to_count.keys())
    if len(levels) > sample_size:
        raise ValueError("Sample size must be >= number of levels")

    shares = {lev: (sample_size * level_to_count[lev]) / total for lev in levels}
    floors = {lev: max(1, int(math.floor(shares[lev]))) for lev in levels}
    allocation = {lev: min(floors[lev], level_to_count[lev]) for lev in levels}
    current = sum(allocation.values())

    if current < sample_size:
        need = sample_size - current
        order = sorted(levels, key=lambda l: shares[l] - allocation[l], reverse=True)
        idx = 0
        while need > 0:
            lev = order[idx % len(levels)]
            if allocation[lev] < level_to_count[lev]:
                allocation[lev] += 1
                need -= 1
            idx += 1
    elif current > sample_size:
        excess = current - sample_size
        order = sorted(levels, key=lambda l: allocation[l], reverse=True)
        idx = 0
        while excess > 0:
            lev = order[idx % len(levels)]
            if allocation[lev] > 1:
                allocation[lev] -= 1
                excess -= 1
            idx += 1

    return allocation


def create_gaia_subset(
    input_path: Path,
    output_path: Path,
    sample_size: int,
    seed: int = 42,
) -> None:
    """
    Create stratified subset by Level proportion.

    Args:
        input_path: Input gaia JSONL.
        output_path: Output subset JSONL path.
        sample_size: Number of samples.
        seed: Random seed.
    """
    random.seed(seed)

    records: List[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))

    level_to_samples: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        level = str(r.get("Level", "")).strip() or "0"
        level_to_samples[level].append(r)

    level_to_count = {k: len(v) for k, v in level_to_samples.items()}
    if sum(level_to_count.values()) < sample_size:
        raise ValueError(
            f"Insufficient samples: {len(records)} < {sample_size}"
        )

    allocation = _compute_proportional_allocation(level_to_count, sample_size)
    selected: List[dict] = []
    for lev, alloc in allocation.items():
        pool = level_to_samples[lev]
        selected.extend(random.sample(pool, k=min(alloc, len(pool))))

    random.shuffle(selected)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dist = Counter(r.get("Level", "") for r in selected)
    print(f"✅ Saved {len(selected)} samples to {output_path}")
    for lev, cnt in sorted(dist.items(), key=lambda x: x[0]):
        print(f"   Level{lev}: {cnt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create GAIA subsets")
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
        help="Output directory (default: subset/)",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[20, 50, 100],
        help="Subset sizes (default: 20 50 100)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir or (script_dir / "subset")
    output_dir.mkdir(parents=True, exist_ok=True)

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

    for size in args.sizes:
        out_path = output_dir / f"gaia_subset_{size}.jsonl"
        create_gaia_subset(input_path, out_path, sample_size=size, seed=args.seed)


if __name__ == "__main__":
    main()
