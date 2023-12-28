# Copyright © 2023 Apple Inc.

import unittest

import mistral
import mlx.core as mx
from mlx.utils import tree_map


class TestMistral(unittest.TestCase):
    def test_model(self):
        vocab_size = 100
        L = 32
        args = mistral.ModelArgs(
            dim=128,
            n_layers=2,
            head_dim=32,
            hidden_dim=256,
            n_heads=4,
            n_kv_heads=4,
            norm_eps=1e-3,
            vocab_size=vocab_size,
        )

        model = mistral.Mistral(args)
        inputs = mx.random.randint(0, vocab_size, (L,))
        logits, cache = model(inputs[None])
        self.assertEqual(logits.shape, [1, L, vocab_size])
        self.assertEqual(logits.dtype, mx.float32)
        self.assertEqual(len(cache), args.n_layers)

        params = tree_map(lambda p: p.astype(mx.float16), model.parameters())
        model.update(params)
        logits, _ = model(inputs[None])
        self.assertEqual(logits.dtype, mx.float16)

    def test_generate(self):
        model, tokenizer = mistral.load_model("mistral-7B-v0.1")
        prompt = mx.array(tokenizer.encode("This is a test"))
        tokens = [t for t, _ in zip(mistral.generate(prompt, model), range(30))]
        mx.eval(tokens)
        tokens = [t.item() for t in tokens]
        expected = [
            302,
            272,
            11843,
            11837,
            1587,
            28723,
            851,
            349,
            865,
            264,
            1369,
            28723,
            13,
            13,
            3381,
            456,
            654,
            264,
            1353,
            11843,
            28725,
            368,
            682,
            347,
            2240,
            767,
            298,
            511,
            28723,
            13,
        ]
        self.assertEqual(tokens, expected)

    def benchmark(self):
        import time

        model, tokenizer = mistral.load_model("mistral-7B-v0.1")
        prompt = mx.random.randint(0, model.vocab_size, (128,))

        # warmup
        for _ in range(2):
            generator = mistral.generate(prompt, model)
            mx.eval(next(generator))

        tic = time.time()
        its = 5
        for _ in range(its):
            generator = mistral.generate(prompt, model)
            mx.eval(next(generator))
        toc = time.time()
        tps = its * prompt.size / (toc - tic)
        print(f"Prompt processing: {tps:.2f} tokens per second")

        # warmup
        for _ in range(2):
            tokens = [t for t, _ in zip(mistral.generate(prompt, model), range(101))]
            mx.eval(tokens)

        time_total = 0.0
        its = 2
        for _ in range(its):
            generator = mistral.generate(prompt, model)
            mx.eval(next(generator))
            tic = time.time()
            tokens = [t for t, _ in zip(generator, range(100))]
            mx.eval(tokens)
            time_total += time.time() - tic

        tps = len(tokens) * its / time_total
        print(f"Token generation: {tps:.3f} tokens per second")


if __name__ == "__main__":
    unittest.main()
