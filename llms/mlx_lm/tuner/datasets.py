import json
from pathlib import Path
from typing import Iterable

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


class TextHFDataset(Dataset):
    """
    A Huggingface dataset for a single blob of text per record
    https://huggingface.co/docs/datasets/en/access
    """

    def __init__(self, hf_dataset, tokenizer, text_feature):
        self._data = hf_dataset
        self._tokenizer = tokenizer
        self._text_feature = text_feature

    def __getitem__(self, idx: int):
        return self._data[idx][self._text_feature]


class CompletionsHFDataset(CompletionsDataset):
    """
    Prompt/completion data from a Huggingface dataset
    https://huggingface.co/docs/datasets/en/access
    """

    def __init__(
        self,
        hf_dataset: Iterable,
        tokenizer: PreTrainedTokenizer,
        prompt_feature: str = None,
        completion_feature: str = None,
    ):
        """

        :param hf_dataset: Iterable HF dataset instance in a 'split' and/or configuration
        :param prompt_feature: The HF dataset 'feature' or column name for the instruction input
        :param completion_feature: The HF dataset 'feature' or column name for the instruction completion
        """
        self._data = hf_dataset
        self._tokenizer = tokenizer
        self._prompt_feature = prompt_feature
        self._completion_feature = completion_feature

    def __getitem__(self, idx: int):
        data = self._data[idx]
        text = self._tokenizer.apply_chat_template(
            [
                {"role": "user", "content": data[self._prompt_feature]},
                {"role": "assistant", "content": data[self._completion_feature]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return text


def load_dataset(args, tokenizer: PreTrainedTokenizer):
    if args.hf_dataset:
        from datasets import load_dataset, get_dataset_infos
        dataset_name = args.hf_dataset["name"]
        print(f"Loading HF dataset {dataset_name}: {get_dataset_infos(dataset_name)}")
        train_split = args.hf_dataset.get("train_split", "train[:80%]")
        valid_split = args.hf_dataset.get("valid_split", "train[-10%:]")
        test_split = args.hf_dataset.get("test_split")
        train = load_dataset(
            dataset_name, args.hf_dataset.get("configuration"), split=train_split
        )
        valid = load_dataset(
            dataset_name, args.hf_dataset.get("configuration"), split=valid_split
        )
        test = (
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
            and prompt_feature in train.features
            and completion_feature
            and completion_feature in train.features
        ):
            train = CompletionsHFDataset(
                train, tokenizer, prompt_feature, completion_feature
            )
            valid = CompletionsHFDataset(
                valid, tokenizer, prompt_feature, completion_feature
            )
            test = (
                CompletionsHFDataset(
                    test, tokenizer, prompt_feature, completion_feature
                )
                if args.test
                else None
            )
        elif text_feature and text_feature in train.features:
            train = TextHFDataset(train, tokenizer, text_feature)
            valid = TextHFDataset(valid, tokenizer, text_feature)
            test = TextHFDataset(test, tokenizer, text_feature) if args.test else None
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
            "Test set not found or empty. Must provide test set for ev aluation."
        )
    return train, valid, test
