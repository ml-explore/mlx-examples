import unittest

import mlx.core as mx
from mlx_lm.sample_utils import min_p_sampling, top_k_sampling, top_p_sampling


class TestSampleUtils(unittest.TestCase):
    def test_top_p_sampling(self):
        probs = mx.array([0.9, 0.0, 0.0, 0.1])[None]
        logits = mx.log(probs)

        actual_logits = top_p_sampling(logits, 0.3)
        actual_probs = mx.softmax(actual_logits.squeeze())
        self.assertEqual(actual_probs.tolist(), [1.0, 0.0, 0.0, 0.0])

        actual_logits = top_p_sampling(logits, 0.95)
        actual_probs = mx.softmax(actual_logits.squeeze())
        self.assertEqual(probs.squeeze().tolist(), actual_probs.tolist())

        probs = mx.array([0.0, 0.5, 0.4, 0.1])[None]
        logits = mx.log(probs)
        actual_logits = top_p_sampling(logits, 0.4)
        actual_probs = mx.softmax(actual_logits.squeeze())
        self.assertEqual(actual_probs.tolist(), [0.0, 1.0, 0.0, 0.0])

        actual_logits = top_p_sampling(logits, 0.6)
        actual_probs = mx.softmax(actual_logits.squeeze())
        self.assertEqual(
            [round(p, 4) for p in actual_probs.tolist()], [0.0, 0.5556, 0.4444, 0.0]
        )

        actual_logits = top_p_sampling(logits, 0.95)
        actual_probs = mx.softmax(actual_logits.squeeze())
        actual_rounded = [round(p, 4) for p in actual_probs.tolist()]
        expected_rounded = [0.0, 0.5, 0.4, 0.1]
        self.assertEqual(actual_rounded, expected_rounded)
        self.assertAlmostEqual(sum(actual_probs.tolist()), 1.0)

        # Batch mode works
        probs = mx.array([[0.9, 0.0, 0.0, 0.1], [0.0, 0.8, 0.1, 0.1]])
        logits = mx.log(probs)
        actual_logits = top_p_sampling(logits, 0.5)
        actual_probs = mx.softmax(actual_logits, axis=-1)
        self.assertEqual(
            actual_probs.tolist(), [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )

    def test_min_p_sampling(self):
        probs = mx.array([0.9, 0.0, 0.0, 0.1])[None]
        logits = mx.log(probs)
        temperature = 1.0
        token = min_p_sampling(logits, 0.8)
        self.assertEqual(token, 0)

        probs = mx.array([0.9, 0.0, 0.0, 0.1])[None]
        logits = mx.log(probs)
        temperature = 1.0
        for _ in range(5):
            token = min_p_sampling(logits, 0.05)
            self.assertTrue(token in (0, 3))

        # Batch mode works
        probs = mx.array([[0.9, 0.0, 0.0, 0.1], [0.0, 0.8, 0.0, 0.1]])
        logits = mx.log(probs)
        tokens = min_p_sampling(logits, 0.7)
        self.assertEqual(tokens.tolist(), [0, 1])

    def test_top_k_sampling(self):
        probs = mx.array([0.9, 0.0, 0.0, 0.1])[None]
        logits = mx.log(probs)

        token = top_k_sampling(logits, 1).item()
        self.assertEqual(token, 0)

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
        self.assertEqual(tokens.tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
