import pandas as pd


def load_hle_dataset(args):
    """
    Load HLE dataset based on subset or category parameters.
    
    Args:
        args: Configuration object with subset, category, and repo_root attributes
        
    Returns:
        pandas.DataFrame: Loaded and processed dataset
        
    Raises:
        ValueError: If both subset and category are null, or both are not null
        FileNotFoundError: If the specified data file doesn't exist
    """
    
    # Validate parameters: exactly one of subset or category must be null
    if (args.subset is None) == (args.category is None):
        raise ValueError("Exactly one of 'subset' or 'category' must be null, not both or neither.")
    
    data_dir = args.repo_root / "data" / "HLE"
    
    # Determine data path based on parameters
    if args.subset is not None:
        # Load based on subset size
        subset_mapping = {
            "small": "hle_subset_50.jsonl",
            "medium": "hle_subset_200.jsonl", 
            "large": "hle_subset_500.jsonl"
        }
        
        if args.subset not in subset_mapping:
            raise ValueError(f"Invalid subset '{args.subset}'. Must be one of: {list(subset_mapping.keys())}")
            
        data_path = data_dir / "subset" / subset_mapping[args.subset]
        print(f"Loading subset: {args.subset} from {data_path}")
        
    else:
        # Load based on category
        category_mapping = {
            "bio": "Biology_Medicine.jsonl",
            "chem": "Chemistry.jsonl", 
            "cs": "Computer_Science_AI.jsonl",
            "engineer": "Engineering.jsonl",
            "social": "Humanities_Social_Science.jsonl",
            "math": "Math.jsonl",
            "physics": "Physics.jsonl",
            "other": "Other.jsonl"
        }
        
        if args.category not in category_mapping:
            raise ValueError(f"Invalid category '{args.category}'. Must be one of: {list(category_mapping.keys())}")
            
        data_path = data_dir / "category" / category_mapping[args.category]
        print(f"Loading category: {args.category} from {data_path}")
    
    # Check if file exists
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load the dataset
    print("Reading:", data_path)
    eval_df = pd.read_json(data_path, lines=True)
    
    # Rename columns as required
    column_mapping = {
        "task_id": "id",
        "Question": "question",
        "Final answer": "answer", 
        "Category": "category"
    }
    
    eval_df = eval_df.rename(columns=column_mapping)
    
    print(f"Loaded {len(eval_df)} samples")
    if 'category' in eval_df.columns:
        print("Category distribution:")
        print(eval_df['category'].value_counts())
    
    return eval_df