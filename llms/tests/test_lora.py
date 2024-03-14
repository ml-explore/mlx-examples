# Copyright © 2024 Apple Inc.

import sys
import unittest
from io import StringIO
from unittest.mock import MagicMock

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import lora, tuner


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

    def test_quantized_print_trainable_parameters(self):
        model = MagicMock()
        model.parameters.return_value = {
            "layer1.weight": MagicMock(size=1e6),
            "layer1.biases": MagicMock(
                size=2e6,
            ),
            "layer1.scales": MagicMock(
                size=2e6,
            ),
            "layer3.weight": MagicMock(size=2e6),
            "lora_a": MagicMock(size=3e6),
            "lora_b": MagicMock(size=4e6),
        }
        model.trainable_parameters.return_value = {
            "layer1.weight": MagicMock(size=1e6),
            "layer3.weight": MagicMock(size=2e6),
        }

        config_8bits = {"quantization": {"bits": 8}}
        expected_output_8bits = "Trainable parameters: 50.000% (3.000M/6.000M)\n"
        lora.print_trainable_parameters(model, config_8bits)
        self.assertEqual(self.capturedOutput.getvalue(), expected_output_8bits)
        self.capturedOutput.truncate(0)
        self.capturedOutput.seek(0)

        config_4bits = {"quantization": {"bits": 4}}
        expected_output_4bits = "Trainable parameters: 30.000% (3.000M/10.000M)\n"
        lora.print_trainable_parameters(model, config_4bits)
        self.assertEqual(self.capturedOutput.getvalue(), expected_output_4bits)
        self.capturedOutput.truncate(0)
        self.capturedOutput.seek(0)

    def test_print_trainable_parameters(self):
        model = MagicMock()
        model.parameters.return_value = {
            "layer1.weight": MagicMock(
                size=4e6,
            ),
            "layer3.weight": MagicMock(
                size=2e6,
            ),
            "lora_a": MagicMock(
                size=3e6,
            ),
            "lora_b": MagicMock(
                size=4e6,
            ),
        }
        model.trainable_parameters.return_value = {
            "layer1.weight": MagicMock(
                size=1e6,
            ),
            "layer3.weight": MagicMock(
                size=2e6,
            ),
        }

        config = {}
        expected_output = "Trainable parameters: 50.000% (3.000M/6.000M)\n"
        lora.print_trainable_parameters(model, config)
        self.assertEqual(self.capturedOutput.getvalue(), expected_output)


if __name__ == "__main__":
    unittest.main()
