import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlx.core as mx
from mlx_lm.gguf import convert_to_gguf


class TestConvertToGGUFWithoutMocks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir_fid = tempfile.TemporaryDirectory()
        cls.test_dir = cls.test_dir_fid.name
        cls.tokenizer_file_path = os.path.join(cls.test_dir, "tokenizer.json")
        with open(cls.tokenizer_file_path, "w") as f:
            f.write("{}")

    @classmethod
    def tearDownClass(cls):
        cls.test_dir_fid.cleanup()

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("mlx.core.save_gguf")
    def test_convert_to_gguf(
        self,
        mock_save_gguf,
        mock_from_pretrained,
    ):
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 3
        mock_tokenizer.get_added_vocab.return_value = {}
        mock_tokenizer.get_vocab.return_value = {"<pad>": 0, "hello": 1, "world": 2}
        mock_tokenizer.all_special_tokens = ["<pad>"]
        mock_tokenizer.all_special_ids = [0]
        mock_from_pretrained.return_value = mock_tokenizer

        model_path = Path(self.test_dir)
        weights = {
            "self_attn.q_proj.weight": mx.random.uniform(shape=[768, 768]),
        }
        config = {
            "num_attention_heads": 1,
            "num_hidden_layers": 1,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "_name_or_path": "test-llama",
        }
        output_file_path = "/fake/output/path/gguf_model.gguf"

        convert_to_gguf(model_path, weights, config, output_file_path)
        called_args, _ = mock_save_gguf.call_args
        self.assertEqual(called_args[0], output_file_path)


if __name__ == "__main__":
    unittest.main()
