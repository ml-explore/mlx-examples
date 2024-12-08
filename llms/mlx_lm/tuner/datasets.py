import json
from pathlib import Path
from typing import Callable, Dict, List, Union

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

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        chat_key: str = "messages",
    ):
        super().__init__(data)
        self._tokenizer = tokenizer
        self._chat_key = chat_key

    def __getitem__(self, idx: int):
        messages = self._data[idx][self._chat_key]
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

    def get_prompt_and_completion(self, idx: int):
        return self._data[idx][self._prompt_key], self._data[idx][self._completion_key]

    def get_item(
        self, idx: int, tokenize: bool = False, add_generation_prompt: bool = True
    ):
        return self._tokenizer.apply_chat_template(
            self._data[idx],
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )

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


class CompletionsDatasetCollection:
    def __init__(self, data: List[Union[ChatDataset, CompletionsDataset]]):
        self.collection = data

    def __fetch_and_process_item__(self, idx: int, handler_fn: Callable):
        iteration = iter(self.collection)
        item = next(iteration)

        curr_idx = idx

        while True:
            try:
                if (curr_idx + 1) <= len(item):
                    return handler_fn(item, curr_idx)
                else:
                    curr_idx -= len(item)
                    item = next(iteration)
            except StopIteration:
                raise IndexError(idx)

    def __getitem__(self, idx: int):
        def getitem(dataset: CompletionsDataset, index: int):
            return dataset[index]

        return self.__fetch_and_process_item__(idx, getitem)

    def get_prompt_and_completion(self, idx: int):
        def getitem(dataset: CompletionsDataset, index: int):
            return dataset.get_prompt_and_completion(index)

        return self.__fetch_and_process_item__(idx, getitem)

    def __len__(self):
        return sum(map(len, self.collection))


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

    def create_hf_dataset(
        dataset_name: Union[None, str],
        text_feature: Union[None, str],
        prompt_feature: Union[None, str],
        completion_feature: Union[None, str],
        chat_feature: Union[None, str],
        split: str = None,
    ):
        ds = datasets.load_dataset(
            dataset_name,
            split=split,
            **hf_args.get("config", {}),
        )
        if prompt_feature and completion_feature:
            return CompletionsDataset(ds, tokenizer, prompt_feature, completion_feature)
        elif chat_feature:
            return ChatDataset(ds, tokenizer, chat_key=chat_feature)
        elif text_feature:
            return Dataset(ds, text_key=text_feature)
        else:
            raise ValueError(
                "Specify either a prompt and completion feature or a text "
                "feature for the Hugging Face dataset."
            )

    def get_hf_custom_features(hf_args):
        return (
            hf_args.get("text_feature"),
            hf_args.get("prompt_feature"),
            hf_args.get("completion_feature"),
            hf_args.get("chat_feature"),
        )

    def get_train_and_valid_splits(hf_args, ds_name):
        train_split = hf_args.get("train_split", "train[:80%]")
        valid_split = hf_args.get("valid_split", "train[-10%:]")
        text_f, prompt_f, completion_f, chat_f = get_hf_custom_features(hf_args)
        train = create_hf_dataset(
            dataset_name=ds_name,
            text_feature=text_f,
            prompt_feature=prompt_f,
            completion_feature=completion_f,
            chat_feature=chat_f,
            split=train_split,
        )
        valid = create_hf_dataset(
            dataset_name=ds_name,
            text_feature=text_f,
            prompt_feature=prompt_f,
            completion_feature=completion_f,
            chat_feature=chat_f,
            split=valid_split,
        )
        return train, valid

    if args.hf_datasets:
        dataset_collection = args.hf_datasets
        train_collection = []
        valid_collection = []
        test_collection = []
        for ds in dataset_collection:
            hf_args = ds["hf_dataset"]
            dataset_name = hf_args["name"]
            print(f"Loading Hugging Face dataset {dataset_name}.")
            text_feature, prompt_feature, completion_feature, chat_f = (
                get_hf_custom_features(hf_args)
            )
            if args.train:
                train, valid = get_train_and_valid_splits(hf_args, dataset_name)
            else:
                train, valid = [], []
            if args.test:
                test = create_hf_dataset(
                    dataset_name=dataset_name,
                    text_feature=text_feature,
                    prompt_feature=prompt_feature,
                    completion_feature=completion_feature,
                    chat_feature=chat_f,
                    split=hf_args.get("test_split"),
                )
            else:
                test = []
            train_collection.append(train)
            valid_collection.append(valid)
            test_collection.append(test)
        return (
            CompletionsDatasetCollection(train_collection),
            CompletionsDatasetCollection(valid_collection),
            CompletionsDatasetCollection(test_collection),
        )
    else:
        hf_args = args.hf_dataset
        dataset_name = hf_args["name"]
        print(f"Loading Hugging Face dataset {dataset_name}.")
        text_feature, prompt_feature, completion_feature, chat_feature = (
            get_hf_custom_features(hf_args)
        )
        if args.train:
            train, valid = get_train_and_valid_splits(hf_args, dataset_name)
        else:
            train, valid = [], []
        if args.test:
            test = create_hf_dataset(
                dataset_name,
                text_feature,
                prompt_feature,
                completion_feature,
                chat_feature,
                split=hf_args.get("test_split"),
            )
        else:
            test = []

    return train, valid, test


def load_dataset(args, tokenizer: PreTrainedTokenizer):
    if getattr(args, "hf_dataset", None) is not None or getattr(args, "hf_datasets"):
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
