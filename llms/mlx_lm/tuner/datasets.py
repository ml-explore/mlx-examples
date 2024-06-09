import json
from pathlib import Path

from transformers import PreTrainedTokenizer


class Dataset:
    """
    Light-weight wrapper to hold lines from a jsonl file or iterable mapping
    """

    def __init__(self, path: Path = None, text_key: str = "text"):
        self._text_key = text_key
        if path:
            with open(path, "r") as fid:
                self._data = [json.loads(l) for l in fid]
        else:
            self._data = None

    def set_data(self, data):
        self._data = data

    def __getitem__(self, idx: int):
        return self._data[idx][self._text_key]

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
    or using user-provided keys for prompt and completion values
    https://platform.openai.com/docs/guides/fine-tuning/example-format
    """

    def __init__(
        self,
        path: Path,
        tokenizer: PreTrainedTokenizer,
        prompt_key: str = "prompt",
        completion_key: str = "completion",
    ):
        super().__init__(path)
        self._tokenizer = tokenizer
        self._prompt_key = prompt_key
        self._completion_key = completion_key

    def __getitem__(self, idx: int):
        data = self._data[idx]
        text = self._tokenizer.apply_chat_template(
            [
                {"role": "user", "content": data[self._prompt_key]},
                {"role": "assistant", "content": data[self._completion_key]},
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
    if args.hf_dataset:
        from datasets import get_dataset_infos, load_dataset

        dataset_name = args.hf_dataset["name"]
        print(f"Loading HF dataset {dataset_name}: {get_dataset_infos(dataset_name)}")
        train_split = args.hf_dataset.get("train_split", "train[:80%]")
        valid_split = args.hf_dataset.get("valid_split", "train[-10%:]")
        test_split = args.hf_dataset.get("test_split")
        train_ds = load_dataset(
            dataset_name, args.hf_dataset.get("configuration"), split=train_split
        )
        valid_ds = load_dataset(
            dataset_name, args.hf_dataset.get("configuration"), split=valid_split
        )
        test_ds = (
            load_dataset(
                dataset_name, args.hf_dataset.get("configuration"), split=test_split
            )
            if args.test
            else None
        )
        text_feature = args.hf_dataset.get("text_feature")
        prompt_feature = args.hf_dataset.get("prompt_feature")
        completion_feature = args.hf_dataset.get("completion_feature")
        if (
            prompt_feature
            and prompt_feature in train_ds.features
            and completion_feature
            and completion_feature in train_ds.features
        ):
            train = CompletionsDataset(
                None, tokenizer, prompt_feature, completion_feature
            )
            train.set_data(train_ds)
            valid = CompletionsDataset(
                None, tokenizer, prompt_feature, completion_feature
            )
            valid.set_data(valid_ds)
            if args.test:
                test = CompletionsDataset(
                    None, tokenizer, prompt_feature, completion_feature
                )
                test.set_data(test_ds)
            else:
                test = None
        elif text_feature and text_feature in train_ds.features:
            train = Dataset(text_key=text_feature)
            train.set_data(train_ds)
            valid = Dataset(text_feature)
            valid.set_data(valid_ds)
            if args.test:
                test = Dataset(text_key=text_feature)
                test.set_data(test_ds)
            else:
                test = None
        else:
            raise ValueError(
                "Need to specify either a prompt and completion feature or a text feature which are "
                "features of the specified HF dataset"
            )

    else:
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
