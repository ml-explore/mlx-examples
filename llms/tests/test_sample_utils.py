import unittest

import mlx.core as mx
from mlx_lm.sample_utils import top_p_sampling


class TestSamplingUtils(unittest.TestCase):
    def test_top_p_sampling(self):
        probs = mx.array([0.9, 0.0, 0.0, 0.1])[None]
        logits = mx.log(probs)
        temperature = 1.0

        token = top_p_sampling(logits, 0.3, temperature).item()
        self.assertEqual(token, 0)

        token = top_p_sampling(logits, 0.95, temperature).item()
        self.assertTrue(token in (0, 3))

        probs = mx.array([0.0, 0.5, 0.4, 0.1])[None]
        logits = mx.log(probs)

        token = top_p_sampling(logits, 0.4, temperature).item()
        self.assertEqual(token, 1)

        token = top_p_sampling(logits, 0.6, temperature).item()
        self.assertTrue(token in (1, 2))

        token = top_p_sampling(logits, 0.95, temperature).item()
        self.assertTrue(token in (1, 2, 3))


if __name__ == "__main__":
    unittest.main()
