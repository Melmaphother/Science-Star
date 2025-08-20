import json
import math
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import os
from datasets import load_dataset


def _compute_proportional_allocation(
    category_to_count: Dict[str, int], sample_size: int
) -> Dict[str, int]:
    """
    Compute per-category allocation using largest remainder method with a minimum
    of 1 per category and a cap at the available count in that category.

    Ensures the total allocated equals sample_size.
    """
    total_available = sum(category_to_count.values())
    categories = list(category_to_count.keys())

    if len(categories) > sample_size:
        raise ValueError("Number of categories exceeds sample size, cannot guarantee at least one sample per category.")

    # Initial allocation: floor of share, but at least 1 and at most available
    shares: Dict[str, float] = {
        cat: (sample_size * category_to_count[cat]) / total_available
        for cat in categories
    }

    floors: Dict[str, int] = {cat: int(math.floor(shares[cat])) for cat in categories}
    remainders: Dict[str, float] = {
        cat: shares[cat] - floors[cat] for cat in categories
    }

    allocation: Dict[str, int] = {}
    for cat in categories:
        proposed = max(1, floors[cat])
        allocation[cat] = min(proposed, category_to_count[cat])

    current_total = sum(allocation.values())

    # If allocated less than sample_size, distribute remaining by largest remainders
    if current_total < sample_size:
        need = sample_size - current_total
        # Prioritize categories with largest remainder, then by larger availability remaining
        order = sorted(
            categories,
            key=lambda c: (remainders[c], category_to_count[c] - allocation[c]),
            reverse=True,
        )
        idx = 0
        while need > 0 and any(
            allocation[c] < category_to_count[c] for c in categories
        ):
            c = order[idx % len(order)]
            if allocation[c] < category_to_count[c]:
                allocation[c] += 1
                need -= 1
            idx += 1

    # If allocated more than sample_size (can happen due to min-1 constraints), reduce
    if sum(allocation.values()) > sample_size:
        excess = sum(allocation.values()) - sample_size
        # Reduce from smallest remainders first, and from categories with largest allocation
        order = sorted(
            categories,
            key=lambda c: (remainders[c], -allocation[c]),
        )
        idx = 0
        while excess > 0:
            c = order[idx % len(order)]
            if allocation[c] > 1:
                allocation[c] -= 1
                excess -= 1
            idx += 1

    # Final guard: do not exceed available counts
    for c in categories:
        if allocation[c] > category_to_count[c]:
            allocation[c] = category_to_count[c]

    # Final sanity check
    total_alloc = sum(allocation.values())
    if total_alloc != sample_size:
        # Best-effort fix: trim or fill arbitrarily while respecting bounds
        if total_alloc > sample_size:
            # Trim from categories with largest allocation
            order = sorted(categories, key=lambda c: allocation[c], reverse=True)
            i = 0
            while sum(allocation.values()) > sample_size and i < len(order):
                c = order[i]
                if allocation[c] > 1:
                    allocation[c] -= 1
                else:
                    i += 1
        elif total_alloc < sample_size:
            order = sorted(
                categories,
                key=lambda c: (category_to_count[c] - allocation[c]),
                reverse=True,
            )
            i = 0
            while sum(allocation.values()) < sample_size and i < len(order):
                c = order[i]
                if allocation[c] < category_to_count[c]:
                    allocation[c] += 1
                else:
                    i += 1

    if sum(allocation.values()) != sample_size:
        raise RuntimeError("Unable to obtain an allocation that satisfies the constraints.")

    return allocation


def create_hle_subset(
    dataset_name: str = "cais/hle",
    split: str = "test",
    output_path: str = "hle_subset.jsonl",
    sample_size: int = 50,
    seed: int = 42,
) -> None:
    """
    Generate HLE subset (50 samples), including only samples where image is an empty string,
    stratified sampling by category proportion, ensuring at least one sample per category.
    Rename/add columns as required and output to JSONL.
    """
    random.seed(seed)

    dataset = load_dataset(dataset_name, split=split)

    # 1) Filter samples where image field is empty string
    filtered_samples: List[dict] = [s for s in dataset if (s.get("image", "") == "")]
    if len(filtered_samples) < sample_size:
        raise ValueError(f"Insufficient samples after filtering: {len(filtered_samples)} < {sample_size}.")

    # 2) Group by category
    category_to_samples: Dict[str, List[dict]] = defaultdict(list)
    for s in filtered_samples:
        category = s.get("category") or "uncategorized"
        category_to_samples[category].append(s)

    # 3) Calculate proportional sampling quota, ensuring at least 1 per category
    category_to_count: Dict[str, int] = {
        k: len(v) for k, v in category_to_samples.items()
    }
    allocation = _compute_proportional_allocation(category_to_count, sample_size)

    # 4) Perform stratified random sampling
    selected: List[dict] = []
    for category, alloc in allocation.items():
        pool = category_to_samples[category]
        if alloc > len(pool):
            # Theoretically shouldn't happen, adding safety guard
            alloc = len(pool)
        selected.extend(random.sample(pool, k=alloc))

    # Randomly shuffle final order to avoid category block clustering
    random.shuffle(selected)

    # 5) Filter out columns with "image" in column name
    output_records = []
    for record in selected:
        filtered_record = {
            key: value for key, value in record.items() 
            if "image" not in key.lower()
        }
        output_records.append(filtered_record)

    # 6) Write to JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for r in output_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Brief output of distribution information
    final_counter = Counter([s.get("category") or "uncategorized" for s in selected])
    print(f"Saved {len(output_records)} records to {output_path}")
    print("Category distribution:")
    for cat, cnt in sorted(final_counter.items(), key=lambda x: x[0]):
        print(f"  {cat}: {cnt}")


if __name__ == "__main__":
    if not os.path.exists("subset"):
        os.makedirs("subset")
    
    create_hle_subset(
        dataset_name="cais/hle",
        split="test",
        output_path="subset/hle_subset_50.jsonl",
        sample_size=50,
        seed=42,
    )
