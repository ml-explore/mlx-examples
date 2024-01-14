import json
import re
import shutil
from pathlib import Path

# Constants copied from transformers library https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L128-L135
SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
ADDED_TOKENS_FILE = "added_tokens.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
FULL_TOKENIZER_FILE = "tokenizer.json"
RE_TOKENIZER_FILE = re.compile(r"tokenizer\.(.*)\.json")

# Hardcoded mapping of model type to vocab file pattern
VOCAB_FILE_MAPPING = {"qwen": "*.tiktoken"}


def copy_tokenizer_files(model_path: Path, mlx_path: Path, model_type: str) -> list:
    """
    Copy specific tokenizer files from a model directory to a specified path.

    Parameters:
    - model_path (Path): Path to the Hugging Face model directory.
    - mlx_path (Path): Destination path for tokenizer files.
    - model_type (str): Type of model to determine specific vocab files.

    Returns:
    - list: List of copied file paths.

    """
    copied_files = []
    auto_map_present = False

    tokenizer_files = [
        SPECIAL_TOKENS_MAP_FILE,
        ADDED_TOKENS_FILE,
        TOKENIZER_CONFIG_FILE,
        FULL_TOKENIZER_FILE,
    ]

    for file_name in tokenizer_files:
        file_path = model_path / file_name
        if file_path.exists():
            shutil.copy(file_path, mlx_path)
            copied_files.append(file_path)
            if file_name == TOKENIZER_CONFIG_FILE:
                with open(file_path, "r") as f:
                    config_data = json.load(f)
                    auto_map_present = "auto_map" in config_data

    for file in model_path.glob("*.json"):
        if RE_TOKENIZER_FILE.match(file.name):
            shutil.copy(file, mlx_path)
            copied_files.append(file)

    if model_type in VOCAB_FILE_MAPPING and auto_map_present:
        vocab_pattern = VOCAB_FILE_MAPPING[model_type]
        for vocab_file in model_path.glob(vocab_pattern):
            shutil.copy(vocab_file, mlx_path)
            copied_files.append(vocab_file)

    return copied_files
