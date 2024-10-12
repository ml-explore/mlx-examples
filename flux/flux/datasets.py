import json
from pathlib import Path

from PIL import Image


class Dataset:
    def __init__(self, flux, args, data):
        self.args = args
        self.flux = flux

        self._data = data

    def __getitem__(self, index: int):
        item = self._data[index]
        image = item['image']
        prompt = item['prompt']

        return image, prompt

    def __len__(self):
        if self._data is None:
            return 0
        return len(self._data)


class LocalDataset(Dataset):

    def __init__(self, flux, args, data_file):
        self.dataset_base = Path(args.dataset)
        with open(data_file, "r") as fid:
            self._data = [json.loads(l) for l in fid]

        super().__init__(flux, args, self._data)

    def __getitem__(self, index: int):
        item = self._data[index]
        image = Image.open(self.dataset_base / item["image"])
        return image, item["prompt"]


class HuggingFaceDataset(Dataset):

    def __init__(self, flux, args):
        from datasets import load_dataset
        df = load_dataset(args.dataset)["train"]
        self._data = df.data
        super().__init__(flux, args, df)

    def __getitem__(self, index: int):
        item = self._data[index]
        return item['image'], item['prompt']


def load_dataset(flux, args):
    dataset_base = Path(args.dataset)
    data_file = dataset_base / "train.jsonl"

    if data_file.exists():
        print(f"Load the local dataset {data_file} .", flush=True)
        # print(f"load local dataset: {data_file}")
        dataset = LocalDataset(flux, args, data_file)
    else:
        print(f"Load the Hugging Face dataset {args.dataset} .", flush=True)
        # print(f"load Hugging Face dataset: {args.dataset}")
        dataset = HuggingFaceDataset(flux, args)

    return dataset
