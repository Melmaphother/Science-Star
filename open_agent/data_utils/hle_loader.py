import pandas as pd
import os


def load_hle_dataset(args):
    data_path = "HLE/hle_subset.jsonl"
    print("Reading:", data_path)
    eval_df = pd.read_json(data_path, lines=True)
    eval_df = eval_df.rename(columns={"Question": "question", "Final answer": "true_answer", "Level": "task"})

    def preprocess_file_paths(row):
        if isinstance(row["file_name"], str) and len(row["file_name"]) > 0:
            row["file_name"] = os.path.join("data", "gaia", args.split, row["file_name"])
        return row

    eval_df = eval_df.apply(preprocess_file_paths, axis=1)
    return eval_df