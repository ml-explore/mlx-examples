# Copyright © 2024 Apple Inc.

import math
import sys
import unittest
from io import StringIO
from unittest.mock import MagicMock

import mlx.nn as nn
import mlx.optimizers as opt
import yaml
from mlx.utils import tree_flatten
from mlx_lm import lora, tuner
from mlx_lm.lora import yaml_loader
from mlx_lm.tuner.lora import LoRALinear
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

    def test_quantized_print_trainable_parameters(self):
        model = MagicMock()
        quantized_linear = MagicMock(spec=nn.QuantizedLinear)
        quantized_linear.weight = MagicMock(size=1e6)
        quantized_linear.bits = 8
        lora_linear = MagicMock(spec=LoRALinear)
        lora_linear.weight = MagicMock(size=2e6)
        lora_linear.parameters.return_value = [lora_linear.weight]

        linear = MagicMock(spec=nn.Linear)
        linear.weight = MagicMock(size=3e6)
        linear.parameters.return_value = [linear.weight]

        model.leaf_modules.return_value = {
            "quantized_linear": quantized_linear,
            "lora_linear": lora_linear,
            "linear": linear,
        }

        model.trainable_parameters.return_value = {
            "layer1.weight": MagicMock(size=1e6),
            "layer3.weight": MagicMock(size=2e6),
        }
        expected_output_8bits = "Trainable parameters: 33.333% (3.000M/9.000M)\n"
        lora.print_trainable_parameters(model)
        self.assertEqual(self.capturedOutput.getvalue(), expected_output_8bits)
        self.capturedOutput.truncate(0)
        self.capturedOutput.seek(0)

        quantized_linear.weight = MagicMock(size=1e6)
        quantized_linear.bits = 4
        expected_output_4bits = "Trainable parameters: 23.077% (3.000M/13.000M)\n"
        lora.print_trainable_parameters(model)
        self.assertEqual(self.capturedOutput.getvalue(), expected_output_4bits)
        self.capturedOutput.truncate(0)
        self.capturedOutput.seek(0)

    def test_print_trainable_parameters(self):
        model = MagicMock()
        linear1 = MagicMock(spec=nn.Linear)
        linear1.weight = MagicMock(size=1e6)
        linear1.parameters.return_value = [linear1.weight]
        linear2 = MagicMock(spec=nn.Linear)
        linear2.weight = MagicMock(size=2e6)
        linear2.parameters.return_value = [linear2.weight]
        lora_linear = MagicMock(spec=LoRALinear)
        lora_linear.weight = MagicMock(size=3e6)
        lora_linear.parameters.return_value = [lora_linear.weight]
        model.leaf_modules.return_value = {
            "linear1": linear1,
            "linear2": linear2,
            "lora_linear": lora_linear,
        }

        model.trainable_parameters.return_value = {
            "layer1.weight": MagicMock(size=1e6),
            "layer3.weight": MagicMock(size=2e6),
        }
        expected_output = "Trainable parameters: 50.000% (3.000M/6.000M)\n"
        lora.print_trainable_parameters(model)
        self.assertEqual(self.capturedOutput.getvalue(), expected_output)


CONFIG_YAML1 = """
schedule:
  name: cosine_decay
  warmup: 100
  arguments: [ 1e-5, 100 ] 
"""

CONFIG_YAML2 = """
schedule:
  warmup: 100
  """

CONFIG_YAML3 = """
schedule:
  name: cosine_decay
"""

CONFIG_YAML4 = """
schedule:
  name: cosine_decay
  arguments: [ 0.1, 10 ]
"""

CONFIG_YAML5 = """
schedule:

"""

CONFIG_YAML6 = """
schedule:
  name: cosine_decay
  warmup: 10
  minimum: 1e-6
  arguments: [ 1e-5, 20 ] 
"""


class TestScheduleConfigs(unittest.TestCase):
    def test_join(self):
        config = yaml.load(CONFIG_YAML1, yaml_loader)
        cos_with_warmup = build_schedule(config["schedule"])
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
        config = yaml.load(CONFIG_YAML4, yaml_loader)
        lr_schedule = build_schedule(config["schedule"])
        lr = lr_schedule(4)
        expected_lr = 0.1 * 0.5 * (1.0 + math.cos(math.pi * 4 / 10))
        self.assertAlmostEqual(lr, expected_lr, delta=1e-7)

    def test_non_zero_warmup(self):
        config = yaml.load(CONFIG_YAML6, yaml_loader)
        lr_schedule = build_schedule(config["schedule"])
        lr = lr_schedule(0)
        self.assertAlmostEqual(lr, 1e-6, delta=1e-7)

    def test_malformed_config(self):
        config = yaml.load(CONFIG_YAML2, yaml_loader)
        self.assertRaises(KeyError, build_schedule, config["schedule"])

        config = yaml.load(CONFIG_YAML3, yaml_loader)
        self.assertRaises(KeyError, build_schedule, config["schedule"])

    def test_empty_config(self):
        config = yaml.load(CONFIG_YAML5, yaml_loader)
        self.assertIsNone(build_schedule(config["schedule"]))


if __name__ == "__main__":
    unittest.main()
