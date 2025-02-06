import pandas as pd
import os

# Define dataset directory
dataset_dir = "/Users/cshang/Desktop/test_grpo/data"

# Convert each Parquet file to JSONL
for file in os.listdir(dataset_dir):
    if file.endswith(".parquet"):
        parquet_path = os.path.join(dataset_dir, file)
        jsonl_path = os.path.join(dataset_dir, file.replace(".parquet", ".jsonl"))
        
        # Load Parquet file
        df = pd.read_parquet(parquet_path)

        # Convert to JSONL format
        df.to_json(jsonl_path, orient="records", lines=True)

        print(f"Converted {parquet_path} -> {jsonl_path}")