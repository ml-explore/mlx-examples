# Copyright © 2024 Apple Inc.

import math
import sys
import unittest
from io import StringIO
from unittest.mock import MagicMock

import mlx.optimizers as opt
from mlx.utils import tree_flatten
from mlx_lm import lora, tuner
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.trainer import evaluate
from mlx_lm.tuner.utils import build_schedule


class TestLora(unittest.TestCase):
    def setUp(self):
        self.capturedOutput = StringIO()
        sys.stdout = self.capturedOutput

    def tearDown(self):
        sys.stdout = sys.__stdout__

    def test_to_lora(self):
        from mlx_lm.models import llama

        args = llama.ModelArgs(
            model_type="llama",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=10_000,
        )

        lora_layers = 4

        def check_config(params):
            n_keys = 2
            if "keys" in params:
                n_keys = len(params["keys"])
            model = llama.Model(args)
            model.freeze()
            tuner.utils.linear_to_lora_layers(model, lora_layers, params)
            trainable_params = sum(
                v.size for _, v in tree_flatten(model.trainable_parameters())
            )
            self.assertEqual(
                trainable_params, lora_layers * params["rank"] * 1024 * 2 * n_keys
            )

        params = {"rank": 8, "alpha": 16, "dropout": 0.0, "scale": 10.0}
        check_config(params)

        params["rank"] = 1
        check_config(params)

        params["keys"] = ["self_attn.k_proj"]
        check_config(params)


class TestScheduleConfig(unittest.TestCase):
    def test_join(self):
        config = {"name": "cosine_decay", "warmup": 100, "arguments": [1e-5, 100]}
        cos_with_warmup = build_schedule(config)
        self.assertIsNotNone(cos_with_warmup)

        self.assertEqual(cos_with_warmup(0), 0.0)
        self.assertAlmostEqual(cos_with_warmup(101), 1e-5, delta=1e-1)
        optimizer = opt.Adam(learning_rate=cos_with_warmup)
        for _ in range(100):
            optimizer.update({}, {})
        self.assertAlmostEqual(optimizer.learning_rate.item(), 1e-5, delta=1e-1)
        for _ in range(100):
            optimizer.update({}, {})
        expected_lr = 1e-5 * 0.5 * (1.0 + math.cos(math.pi * 200 / 10))
        self.assertAlmostEqual(optimizer.learning_rate.item(), expected_lr, delta=1e-1)

    def test_single_schedule(self):

        config = {
            "name": "cosine_decay",
            "arguments": [0.1, 10],
        }
        lr_schedule = build_schedule(config)
        lr = lr_schedule(4)
        expected_lr = 0.1 * 0.5 * (1.0 + math.cos(math.pi * 4 / 10))
        self.assertAlmostEqual(lr, expected_lr, delta=1e-7)

    def test_non_zero_warmup(self):
        config = {
            "name": "cosine_decay",
            "warmup": 10,
            "warmup_init": 1e-6,
            "arguments": [1e-5, 20],
        }
        lr_schedule = build_schedule(config)
        lr = lr_schedule(0)
        self.assertAlmostEqual(lr, 1e-6, delta=1e-7)

    def test_malformed_config(self):
        config = {"warmup": 100}
        self.assertRaises(KeyError, build_schedule, config)

        config = {"cosine_decay": None}
        self.assertRaises(KeyError, build_schedule, config)

    def test_evaluate_calls(self):
        mock_model = MagicMock()
        mock_dataset = MagicMock()
        mock_tokenizer = MagicMock()
        mock_default_loss = MagicMock()
        mock_iterate_batches = MagicMock()

        mock_iterate_batches.return_value = [
            (MagicMock(), MagicMock()),
            (MagicMock(), MagicMock()),
            (MagicMock(), MagicMock()),
            (MagicMock(), MagicMock()),
            (MagicMock(), MagicMock()),
        ]

        mock_default_loss.side_effect = [
            (MagicMock(return_value=0.5), MagicMock(return_value=100)),
            (MagicMock(return_value=0.3), MagicMock(return_value=200)),
            (MagicMock(return_value=0.2), MagicMock(return_value=150)),
            (MagicMock(return_value=0.4), MagicMock(return_value=180)),
            (MagicMock(return_value=0.6), MagicMock(return_value=120)),
        ]
        evaluate(
            model=mock_model,
            dataset=mock_dataset,
            tokenizer=mock_tokenizer,
            batch_size=2,
            num_batches=2,
            max_seq_length=2048,
            loss=mock_default_loss,
            iterate_batches=mock_iterate_batches,
        )

        mock_iterate_batches.assert_called_once_with(
            dataset=mock_dataset,
            tokenizer=mock_tokenizer,
            batch_size=2,
            max_seq_length=2048,
        )
        self.assertEqual(mock_default_loss.call_count, 2)

    def test_evaluate_infinite_batches(self):
        mock_model = MagicMock()
        mock_dataset = MagicMock()
        mock_tokenizer = MagicMock()
        mock_default_loss = MagicMock()
        mock_iterate_batches = MagicMock()

        mock_iterate_batches.return_value = [
            (MagicMock(), MagicMock()),
            (MagicMock(), MagicMock()),
            (MagicMock(), MagicMock()),
        ]

        mock_default_loss.side_effect = [
            (MagicMock(return_value=0.5), MagicMock(return_value=100)),
            (MagicMock(return_value=0.3), MagicMock(return_value=200)),
            (MagicMock(return_value=0.2), MagicMock(return_value=150)),
        ]

        evaluate(
            model=mock_model,
            dataset=mock_dataset,
            tokenizer=mock_tokenizer,
            batch_size=2,
            num_batches=-1,
            max_seq_length=2048,
            loss=mock_default_loss,
            iterate_batches=mock_iterate_batches,
        )

        mock_iterate_batches.assert_called_once_with(
            dataset=mock_dataset,
            tokenizer=mock_tokenizer,
            batch_size=2,
            max_seq_length=2048,
        )
        self.assertEqual(mock_default_loss.call_count, 3)


if __name__ == "__main__":
    unittest.main()
