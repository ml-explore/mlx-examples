import json
from pathlib import Path

from PIL import Image


class Dataset:
    def __init__(self, data):
        self._data = data

    def __len__(self):
        if self._data is None:
            return 0
        return len(self._data)


class LocalDataset(Dataset):

    def __init__(self, dataset: str, data_file):
        self.dataset_base = Path(dataset)
        with open(data_file, "r") as fid:
            self._data = [json.loads(l) for l in fid]

        super().__init__(self._data)

    def __getitem__(self, index: int):
        item = self._data[index]
        image = Image.open(self.dataset_base / item["image"])
        return image, item["prompt"]


class HuggingFaceDataset(Dataset):

    def __init__(self, dataset: str):
        from datasets import load_dataset

        df = load_dataset(dataset)["train"]
        self._data = df.data
        super().__init__(df)

    def __getitem__(self, index: int):
        item = self._data[index]
        return item["image"], item["prompt"]


def load_dataset(dataset: str):
    dataset_base = Path(dataset)
    data_file = dataset_base / "train.jsonl"

    if data_file.exists():
        print(f"Load the local dataset {data_file} .", flush=True)
        dataset = LocalDataset(dataset, data_file)
    else:
        print(f"Load the Hugging Face dataset {dataset} .", flush=True)
        dataset = HuggingFaceDataset(dataset)

    return dataset
