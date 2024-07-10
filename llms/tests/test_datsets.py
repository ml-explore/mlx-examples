# Copyright © 2024 Apple Inc.

import json
import os
import tempfile
import types
import unittest

from mlx_lm.tuner import datasets
from transformers import AutoTokenizer

HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"


class TestDatasets(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir_fid = tempfile.TemporaryDirectory()
        cls.test_dir = cls.test_dir_fid.name
        if not os.path.isdir(cls.test_dir):
            os.mkdir(cls.test_dir_fid.name)

    @classmethod
    def tearDownClass(cls):
        cls.test_dir_fid.cleanup()

    def save_data(self, data):
        for ds in ["train", "valid"]:
            with open(os.path.join(self.test_dir, f"{ds}.jsonl"), "w") as fid:
                for l in data:
                    json.dump(l, fid)
                    fid.write("\n")

    def test_text(self):
        data = {"text": "This is an example for the model."}
        self.save_data(4 * [data])
        args = types.SimpleNamespace(train=True, test=False, data=self.test_dir)
        train, valid, test = datasets.load_dataset(args, None)
        self.assertEqual(len(train), 4)
        self.assertEqual(len(valid), 4)
        self.assertEqual(len(test), 0)
        self.assertTrue(len(train[0]) > 0)
        self.assertTrue(len(valid[0]) > 0)
        self.assertTrue(isinstance(train, datasets.Dataset))

    def test_completions(self):
        data = {"prompt": "What is the capital of France?", "completion": "Paris."}
        self.save_data(4 * [data])
        args = types.SimpleNamespace(train=True, test=False, data=self.test_dir)
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH)
        train, valid, test = datasets.load_dataset(args, tokenizer)
        self.assertEqual(len(train), 4)
        self.assertEqual(len(valid), 4)
        self.assertEqual(len(test), 0)
        self.assertTrue(len(train[0]) > 0)
        self.assertTrue(len(valid[0]) > 0)
        self.assertTrue(isinstance(train, datasets.CompletionsDataset))

    def test_chat(self):
        data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello."},
                {"role": "assistant", "content": "How can I assistant you today."},
            ]
        }
        self.save_data(4 * [data])
        args = types.SimpleNamespace(train=True, test=False, data=self.test_dir)
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH)
        train, valid, test = datasets.load_dataset(args, tokenizer)
        self.assertEqual(len(train), 4)
        self.assertEqual(len(valid), 4)
        self.assertEqual(len(test), 0)
        self.assertTrue(len(train[0]) > 0)
        self.assertTrue(len(valid[0]) > 0)
        self.assertTrue(isinstance(train, datasets.ChatDataset))

    def test_hf(self):
        args = types.SimpleNamespace(
            hf_dataset={
                "name": "billsum",
                "prompt_feature": "text",
                "completion_feature": "summary",
            },
            test=False,
            train=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH)
        train, valid, test = datasets.load_dataset(args, tokenizer)
        self.assertTrue(len(train) > 0)
        self.assertTrue(len(train[0]) > 0)
        self.assertTrue(len(valid) > 0)
        self.assertTrue(len(valid[0]) > 0)
        self.assertEqual(len(test), 0)


if __name__ == "__main__":
    unittest.main()
