import unittest

import mlx.core as mx
from mlx_lm.sample_utils import min_p_sampling, top_k_sampling, top_p_sampling


class TestSampleUtils(unittest.TestCase):
    def test_top_p_sampling(self):
        probs = mx.array([0.9, 0.0, 0.0, 0.1])[None]
        logits = mx.log(probs)
        temperature = 1.0

        token = top_p_sampling(logits, 0.3, temperature)
        self.assertEqual(token.shape, (1,))
        self.assertEqual(token.item(), 0)

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

        # Batch mode works
        probs = mx.array([[0.9, 0.0, 0.0, 0.1], [0.0, 0.5, 0.4, 0.1]])
        logits = mx.log(probs)
        token = top_p_sampling(logits, 0.4, temperature)
        self.assertEqual(token.shape, (2,))
        self.assertEqual(token.tolist(), [0, 1])

    def test_min_p_sampling(self):
        probs = mx.array([0.9, 0.0, 0.0, 0.1])[None]
        logits = mx.log(probs)
        token = min_p_sampling(logits, 0.8)
        self.assertEqual(token.shape, (1,))
        self.assertEqual(token.item(), 0)

        probs = mx.array([0.9, 0.0, 0.0, 0.1])[None]
        logits = mx.log(probs)
        for _ in range(5):
            token = min_p_sampling(logits, 0.05)
            self.assertTrue(token in (0, 3))

        # Batch mode works
        probs = mx.array([[0.6, 0.0, 0.0, 0.4], [0.7, 0.0, 0.0, 0.3]])
        logits = mx.log(probs)
        for _ in range(5):
            token = min_p_sampling(logits, 0.65)
            self.assertEqual(token.shape, (2,))
            self.assertTrue(token.tolist() in ([0, 0], [3, 0]))

    def test_top_k_sampling(self):
        probs = mx.array([0.9, 0.0, 0.0, 0.1])[None]
        logits = mx.log(probs)

        token = top_k_sampling(logits, 1)
        self.assertEqual(token.shape, (1,))
        self.assertEqual(token.item(), 0)

        probs = mx.array([0.5, 0.0, 0.0, 0.5])[None]
        tokens = set()
        for _ in range(100):
            token = top_k_sampling(logits, 2)
            tokens.add(token.item())
        self.assertEqual(tokens, {0, 3})

        # Batch mode works
        probs = mx.array([[0.9, 0.0, 0.0, 0.1], [0.0, 0.8, 0.0, 0.1]])
        logits = mx.log(probs)

        tokens = top_k_sampling(logits, 1)
        self.assertEqual(tokens.shape, (2,))
        self.assertEqual(tokens.tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
