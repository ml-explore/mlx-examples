import json
from pathlib import Path

from transformers import PreTrainedTokenizer


class Dataset:
    """
    Light-weight wrapper to hold lines from a jsonl file
    """

    def __init__(self, path: Path):
        with open(path, "r") as fid:
            self._data = [json.loads(l) for l in fid]

    def __getitem__(self, idx: int):
        return self._data[idx]["text"]

    def __len__(self):
        if self._data is None:
            return 0
        return len(self._data)


class ChatDataset(Dataset):
    """
    A dataset for chat data in the format of {"messages": [...]}
    https://platform.openai.com/docs/guides/fine-tuning/example-format
    """

    def __init__(self, path: Path, tokenizer: PreTrainedTokenizer):
        super().__init__(path)
        self._tokenizer = tokenizer

    def __getitem__(self, idx: int):
        messages = self._data[idx]["messages"]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return text


class CompletionsDataset(Dataset):
    """
    A dataset for prompt-completion data in the format of {"prompt": ..., "completion": ...}
    https://platform.openai.com/docs/guides/fine-tuning/example-format
    """

    def __init__(self, path: Path, tokenizer: PreTrainedTokenizer):
        super().__init__(path)
        self._tokenizer = tokenizer

    def __getitem__(self, idx: int):
        data = self._data[idx]
        text = self._tokenizer.apply_chat_template(
            [
                {"role": "user", "content": data["prompt"]},
                {"role": "assistant", "content": data["completion"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return text


def create_dataset(path: Path, tokenizer: PreTrainedTokenizer = None):
    # Return empty dataset for non-existent paths
    if not path.exists():
        return []
    with open(path, "r") as fid:
        first_line = next(fid)
        first_obj = json.loads(first_line)
    if "messages" in first_obj:
        return ChatDataset(path, tokenizer)
    elif "prompt" in first_obj and "completion" in first_obj:
        return CompletionsDataset(path, tokenizer)
    elif "text" in first_obj:
        return Dataset(path)
    else:
        raise ValueError(
            "Unsupported data format, check the supported formats here:\n"
            "https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md#data."
        )


def load_dataset(args, tokenizer: PreTrainedTokenizer):
    names = ("train", "valid", "test")
    data_path = Path(args.data)
    train, valid, test = [
        create_dataset(data_path / f"{n}.jsonl", tokenizer) for n in names
    ]
    if args.train and len(train) == 0:
        raise ValueError(
            "Training set not found or empty. Must provide training set for fine-tuning."
        )
    if args.train and len(valid) == 0:
        raise ValueError(
            "Validation set not found or empty. Must provide validation set for fine-tuning."
        )
    if args.test and len(test) == 0:
        raise ValueError(
            "Test set not found or empty. Must provide test set for evaluation."
        )
    return train, valid, test
