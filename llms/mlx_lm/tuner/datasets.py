import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer


class GRPODataset:
    """
    Dataset wrapper for GRPO training data.
    Each example should have a 'prompt' and 'answer' field.
    Returns data in (prompt_tokens, answer_tokens, prompt_str, answer_str) tuple format.
    """
    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        use_chat_template: bool = False,
        use_prompt: bool = False
    ):
        self._data = []
        for item in data:
            prompt_str = str(item[prompt_key])
            answer_str = str(item[answer_key])
            if use_chat_template:
                prompt_tokens = tokenizer.apply_chat_template(
                    [
                        {'role': 'system', 'content': """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
                The assistantfirst thinks about the reasoning process in the mind and then provides the user with the answer.
                The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""},
                    {'role': 'user', 'content': prompt_str}
                    ],
                )
                answer_tokens = tokenizer.encode(answer_str)
            else:
                if use_prompt:
                    prompt_tokens = tokenizer.encode(f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
            The assistantfirst thinks about the reasoning process in the mind and then provides the user with the answer.
            The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.
            User: {prompt_str}. Assistant: """)
                else:
                    prompt_tokens = tokenizer.encode(prompt_str)
                answer_tokens = tokenizer.encode(answer_str)
            self._data.append((prompt_tokens, answer_tokens, prompt_str, answer_str))

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int], str, str]:
        """Returns a (prompt_tokens, answer_tokens, prompt_str, answer_str) tuple."""
        return self._data[idx]

    def __len__(self) -> int:
        """Returns the number of examples in the dataset."""
        return len(self._data)


class Dataset:
    """
    Light-weight wrapper to hold a dataset.
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        text_key: str = "text",
    ):
        self._data = [tokenizer.encode(d[text_key]) for d in data]
        for d in self._data:
            if d[-1] != tokenizer.eos_token_id:
                d.append(tokenizer.eos_token_id)

    def __getitem__(self, idx: int):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


class ChatDataset:
    """
    A dataset for chat data in the format of {"messages": [...]}
    https://platform.openai.com/docs/guides/fine-tuning/example-format
    """

    def __init__(self, data: List[Dict[str, str]], tokenizer: PreTrainedTokenizer):
        self._data = [
            tokenizer.apply_chat_template(
                d["messages"],
                tools=d.get("tools", None),
            )
            for d in data
        ]

    def __getitem__(self, idx: int):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


class CompletionsDataset:
    """
    A dataset for prompt-completion data in the format of {"prompt": ..., "completion": ...}
    or using user-provided keys for prompt and completion values
    https://platform.openai.com/docs/guides/fine-tuning/example-format
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        prompt_key: str,
        completion_key: str,
    ):
        self._data = [
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": d[prompt_key]},
                    {"role": "assistant", "content": d[completion_key]},
                ],
            )
            for d in data
        ]

    def __getitem__(self, idx: int):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


def create_dataset(
    args,
    data,
    tokenizer: PreTrainedTokenizer,
    prompt_feature: Optional[str] = None,
    completion_feature: Optional[str] = None,
):
    prompt_feature = prompt_feature or "prompt"
    completion_feature = completion_feature or "completion"
    sample = data[0]

    if args.training_mode == "normal":
        if "messages" in sample:
            return ChatDataset(data, tokenizer)
        elif prompt_feature in sample and completion_feature in sample:
            return CompletionsDataset(data, tokenizer, prompt_feature, completion_feature)
        elif "text" in sample:
            return Dataset(data, tokenizer)
        else:
            raise ValueError(
                "Unsupported data format, check the supported formats here:\n"
                "https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md#data."
            )
    else:
        return GRPODataset(
            data=data,
            tokenizer=tokenizer,
            prompt_key="prompt",
            answer_key="answer",
            use_chat_template=args.use_chat_template,
            use_prompt=args.use_prompt
        )


def load_local_dataset(
    args,
    data_path: Path,
    tokenizer: PreTrainedTokenizer,
    prompt_feature: Optional[str] = None,
    completion_feature: Optional[str] = None,
):
    def load_subset(path):
        if not path.exists():
            return []
        with open(path, "r") as fid:
            data = [json.loads(l) for l in fid]
        return create_dataset(args, data, tokenizer, prompt_feature, completion_feature)

    names = ("train", "valid", "test")
    train, valid, test = [load_subset(data_path / f"{n}.jsonl") for n in names]
    return train, valid, test


def load_hf_dataset(
    args,
    data_id: str,
    tokenizer: PreTrainedTokenizer,
    prompt_feature: Optional[str] = None,
    completion_feature: Optional[str] = None,
):
    from datasets import exceptions, load_dataset

    try:
        dataset = load_dataset(data_id)

        names = ("train", "valid", "test")

        train, valid, test = [
            (
                create_dataset(
                    args, dataset[n], tokenizer, prompt_feature, completion_feature
                )
                if n in dataset.keys()
                else []
            )
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
            return Dataset(ds, tokenizer, text_key=text_feature)
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
    if getattr(args, "hf_dataset", False):
        train, valid, test = load_custom_hf_dataset(args, tokenizer)
    else:
        data_path = Path(args.data)

        prompt_feature = getattr(args, "prompt_feature", None)
        completion_feature = getattr(args, "completion_feature", None)
        if data_path.exists():
            train, valid, test = load_local_dataset(
                args, data_path, tokenizer, prompt_feature, completion_feature
            )
        else:
            print(f"Loading Hugging Face dataset {args.data}.")
            train, valid, test = load_hf_dataset(
                args, args.data, tokenizer, prompt_feature, completion_feature
            )

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
