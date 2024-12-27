# Copyright © 2024 Apple Inc.

import unittest

from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.utils import generate, batch_generate, load


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

    def test_generate_with_logit_bias(self):
        logit_bias = {0: 2000.0, 1: -20.0}
        text = generate(
            self.model,
            self.tokenizer,
            "hello",
            max_tokens=5,
            logits_processors=make_logits_processors(logit_bias),
            verbose=False,
        )
        self.assertEqual(text, "!!!!!")

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
            logits_processors=[logits_processor],
        )
        self.assertEqual(all_toks.shape[-1], len(init_toks) + 5)

    def test_batch_generate(self):
        logit_bias = {0: 20.0, 1: -20.0}
        texts = batch_generate(
            self.model,
            self.tokenizer,
            [
                "hello",
                "this is a longer prompt to test out the padding and masking. hello",
            ],
            max_tokens=5,
            prefill_step_size=4,
            sampler=make_sampler(temp=1.0, min_p=0.1),
            logits_processors=make_logits_processors(logit_bias, repetition_penalty=2.0),
            verbose=False,
        )
        self.assertEqual(texts, ['!', '!'])


if __name__ == "__main__":
    unittest.main()
