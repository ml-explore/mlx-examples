import unittest
from transformers import PreTrainedTokenizerFast
from llms.mlx_lm.tuner.datasets import CompletionsDataset, create_dataset

class TestCompletionsDataset(unittest.TestCase):

    def setUp(self):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("gpt2")
        self.data = [
            {"prompt": "What is the capital of France?", "completion": "Paris."},
            {"prompt": "What is the capital of Germany?", "completion": "Berlin."}
        ]

    def test_completions_dataset(self):
        dataset = CompletionsDataset(self.data, self.tokenizer, "prompt", "completion")
        self.assertEqual(len(dataset), 2)
        self.assertTrue(isinstance(dataset[0], list))
        self.assertTrue(isinstance(dataset[1], list))

class TestCreateDataset(unittest.TestCase):

    def setUp(self):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("gpt2")
        self.data_completions = [
            {"prompt": "What is the capital of France?", "completion": "Paris."},
            {"prompt": "What is the capital of Germany?", "completion": "Berlin."}
        ]
        self.data_text = [
            {"text": "This is a sample text."},
            {"text": "This is another sample text."}
        ]
        self.data_chat = [
            {"messages": [{"role": "user", "content": "Hello."}, {"role": "assistant", "content": "Hi there!"}]}
        ]

    def test_create_completions_dataset(self):
        dataset = create_dataset(self.data_completions, self.tokenizer, "prompt", "completion")
        self.assertEqual(len(dataset), 2)
        self.assertTrue(isinstance(dataset[0], list))
        self.assertTrue(isinstance(dataset[1], list))

    def test_create_text_dataset(self):
        dataset = create_dataset(self.data_text, self.tokenizer)
        self.assertEqual(len(dataset), 2)
        self.assertTrue(isinstance(dataset[0], list))
        self.assertTrue(isinstance(dataset[1], list))

    def test_create_chat_dataset(self):
        dataset = create_dataset(self.data_chat, self.tokenizer)
        self.assertEqual(len(dataset), 1)
        self.assertTrue(isinstance(dataset[0], list))

if __name__ == "__main__":
    unittest.main()
