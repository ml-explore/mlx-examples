# Copyright © 2024 Apple Inc.

import unittest

from mlx_lm.utils import generate, load


class TestGenerate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"
        cls.model, cls.tokenizer = load(HF_MODEL_PATH)

    def test_generate(self):
        # Simple test that generation runs
        text = generate(
            self.model, self.tokenizer, "hello", max_tokens=5, verbose=False
        )

    def test_generate_with_processor(self):
        init_toks = self.tokenizer.encode("hello")

        all_toks = None

        def logits_processor(toks, logits):
            nonlocal all_toks
            all_toks = toks
            return logits

        generate(
            self.model,
            self.tokenizer,
            "hello",
            max_tokens=5,
            verbose=False,
            logits_processor=logits_processor,
        )
        self.assertEqual(len(all_toks), len(init_toks) + 5)


if __name__ == "__main__":
    unittest.main()
