import json
from pathlib import Path
from typing import Dict, List

from transformers import PreTrainedTokenizer


class Dataset:
    """
    Light-weight wrapper to hold a dataset.
    """

    def __init__(self, data: List[Dict[str, str]], text_key: str = "text"):
        self._text_key = text_key
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

    def __init__(self, data: List[Dict[str, str]], tokenizer: PreTrainedTokenizer):
        super().__init__(data)
        self._tokenizer = tokenizer

    def __getitem__(self, idx: int):
        messages = self._data[idx]["messages"]
        text = self._tokenizer.apply_chat_template(
            messages,
            tools=self._data[idx].get("tools", None),
            tokenize=False,
            add_generation_prompt=True,
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
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        prompt_key: str = "prompt",
        completion_key: str = "completion",
    ):
        super().__init__(data)
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


def create_dataset(data, tokenizer: PreTrainedTokenizer = None):
    sample = data[0]

    if "messages" in sample:
        return ChatDataset(data, tokenizer)
    elif "prompt" in sample and "completion" in sample:
        return CompletionsDataset(data, tokenizer)
    elif "text" in sample:
        return Dataset(data)
    else:
        raise ValueError(
            "Unsupported data format, check the supported formats here:\n"
            "https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md#data."
        )


def load_local_dataset(data_path: Path, tokenizer: PreTrainedTokenizer):
    def load_subset(path):
        if not path.exists():
            return []
        with open(path, "r") as fid:
            data = [json.loads(l) for l in fid]
        return create_dataset(data, tokenizer)

    names = ("train", "valid", "test")
    train, valid, test = [load_subset(data_path / f"{n}.jsonl") for n in names]
    return train, valid, test


def load_hf_dataset(data_id: str, tokenizer: PreTrainedTokenizer):
    from datasets import exceptions, load_dataset

    try:
        dataset = load_dataset(data_id)

        names = ("train", "valid", "test")

        train, valid, test = [
            create_dataset(dataset[n], tokenizer) if n in dataset.keys() else []
            for n in names
        ]

    except exceptions.DatasetNotFoundError:
        raise ValueError(f"Not found Hugging Face dataset: {data_id} .")

    return train, valid, test


def load_custom_hf_dataset(args, tokenizer: PreTrainedTokenizer):
    import datasets

    hf_args = args.hf_dataset
    dataset_name = hf_args["name"]
    print(f"Loading Hugging Face dataset {dataset_name}.")
    text_feature = hf_args.get("text_feature")
    prompt_feature = hf_args.get("prompt_feature")
    completion_feature = hf_args.get("completion_feature")

    def create_hf_dataset(split: str = None):
        ds = datasets.load_dataset(
            dataset_name,
            split=split,
            **hf_args.get("config", {}),
        )
        if prompt_feature and completion_feature:
            return CompletionsDataset(ds, tokenizer, prompt_feature, completion_feature)
        elif text_feature:
            return Dataset(train_ds, text_key=text_feature)
        else:
            raise ValueError(
                "Specify either a prompt and completion feature or a text "
                "feature for the Hugging Face dataset."
            )

    if args.train:
        train_split = hf_args.get("train_split", "train[:80%]")
        valid_split = hf_args.get("valid_split", "train[-10%:]")
        train = create_hf_dataset(split=train_split)
        valid = create_hf_dataset(split=valid_split)
    else:
        train, valid = [], []
    if args.test:
        test = create_hf_dataset(split=hf_args.get("test_split"))
    else:
        test = []

    return train, valid, test


def load_dataset(args, tokenizer: PreTrainedTokenizer):
    if getattr(args, "hf_dataset", None) is not None:
        train, valid, test = load_custom_hf_dataset(args, tokenizer)
    else:
        data_path = Path(args.data)
        if data_path.exists():
            train, valid, test = load_local_dataset(data_path, tokenizer)
        else:
            print(f"Loading Hugging Face dataset {args.data}.")
            train, valid, test = load_hf_dataset(args.data, tokenizer)

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
