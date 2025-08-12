import pandas as pd
import os

def load_hle_dataset(args):
    data_path = args.repo_root / "data" / "HLE" / "hle_subset.jsonl"
    print("Reading:", data_path)
    eval_df = pd.read_json(data_path, lines=True)
    eval_df = eval_df.rename(columns={"Question": "question", "Final answer": "true_answer", "Level": "task"})

    return eval_df