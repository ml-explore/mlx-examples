# Copyright © 2024 Apple Inc.

import unittest

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import utils

HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"


class TestUtils(unittest.TestCase):

    def test_load(self):
        model, _ = utils.load(HF_MODEL_PATH)

        model_lazy, _ = utils.load(HF_MODEL_PATH, lazy=True)

        mx.eval(model_lazy.parameters())

        p1 = model.layers[0].mlp.up_proj.weight
        p2 = model_lazy.layers[0].mlp.up_proj.weight
        self.assertTrue(mx.allclose(p1, p2))

    def test_make_shards(self):
        from mlx_lm.models import llama

        args = llama.ModelArgs(
            model_type="llama",
            hidden_size=2048,
            num_hidden_layers=32,
            intermediate_size=4096,
            num_attention_heads=32,
            rms_norm_eps=1e-5,
            vocab_size=30_000,
        )
        model = llama.Model(args)
        weights = tree_flatten(model.parameters())
        gb = sum(p.nbytes for _, p in weights) // 2**30
        shards = utils.make_shards(dict(weights), 1)
        self.assertTrue(gb <= len(shards) <= gb + 1)


if __name__ == "__main__":
    unittest.main()
