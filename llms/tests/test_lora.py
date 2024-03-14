# Copyright © 2024 Apple Inc.

import unittest

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import tuner, utils


class TestLora(unittest.TestCase):

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


if __name__ == "__main__":
    unittest.main()
