import os
import json
from datasets import load_dataset
from tqdm import tqdm

def process_and_save_by_category(dataset_name: str, split: str, output_dir: str = 'category'):
    """
    Loads a Hugging Face dataset, groups it by the 'Category' column,
    and saves selected columns into separate .jsonl files.

    Args:
        dataset_name (str): The name of the dataset on the Hugging Face Hub.
        split (str): The dataset split to process (e.g., 'train', 'test').
        output_dir (str): The directory where the output files will be saved.
    """
    print(f"Loading dataset '{dataset_name}' split '{split}'...")
    
    try:
        # Load the specified split of the dataset
        # The 'HuggingFaceH4/hle' dataset might require a specific configuration,
        # for this example, we'll assume it loads directly.
        # Please adjust if your dataset 'hle' has a different loading mechanism.
        dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure the dataset name and split are correct.")
        return

    print("Dataset loaded successfully.")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory '{output_dir}' is ready.")

    # A dictionary to hold file handles for each category
    # This is memory-efficient as we don't load the whole dataset into memory
    file_handlers = {}

    try:
        # Iterate over the dataset with a progress bar
        for sample in tqdm(dataset, desc="Processing samples"):
            # 1) 过滤 image 字段不为空的样本
            if sample.get("image", "") == "":
                continue  # 跳过 image 为空的样本
            
            # Get the category for the current sample
            category = sample.get('category')
            if not category:
                # Handle cases where 'Category' might be missing or empty
                category = 'uncategorized'

            # Sanitize category name to be used as a filename
            # Replace spaces and slashes to prevent path issues
            safe_category_name = category.replace(' ', '_').replace('/', '_')

            # Check if we already have an open file for this category
            if safe_category_name not in file_handlers:
                file_path = os.path.join(output_dir, f"{safe_category_name}.jsonl")
                # Open the file in append mode with utf-8 encoding
                file_handlers[safe_category_name] = open(file_path, 'a', encoding='utf-8')

            # 2) 保存所有原始列，但排除包含 "image" 的列
            record = {
                key: value for key, value in sample.items() 
                if "image" not in key.lower()
            }

            # Write the JSON record as a single line in the corresponding file
            file_handlers[safe_category_name].write(json.dumps(record, ensure_ascii=False) + '\n')

    finally:
        # --- Crucial Step: Close all open files ---
        print("\nClosing all file handlers...")
        for handler in file_handlers.values():
            handler.close()
        print("Processing complete.")

if __name__ == '__main__':
    # --- Configuration ---
    # Replace with the actual name of your dataset on Hugging Face Hub
    DATASET_NAME = './hle' 
    # Specify the split you want to process, e.g., 'train' or 'test'
    DATASET_SPLIT = 'test'
    
    process_and_save_by_category(dataset_name=DATASET_NAME, split=DATASET_SPLIT)