import unittest

import mlx.core as mx
from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p


class TestSampleUtils(unittest.TestCase):
    def test_apply_top_p(self):
        probs = mx.array([0.9, 0.0, 0.0, 0.1])[None]
        logits = mx.log(probs)

        new_logits = apply_top_p(logits, 0.3)
        actual_probs = mx.softmax(new_logits.squeeze())
        self.assertEqual(actual_probs.tolist(), [1.0, 0.0, 0.0, 0.0])

        new_logits = apply_top_p(logits, 0.95)
        actual_probs = mx.softmax(new_logits.squeeze())
        self.assertTrue(mx.allclose(probs.squeeze(), actual_probs))

        probs = mx.array([0.0, 0.5, 0.4, 0.1])[None]
        logits = mx.log(probs)
        new_logits = apply_top_p(logits, 0.4)
        actual_probs = mx.softmax(new_logits.squeeze())
        self.assertEqual(actual_probs.tolist(), [0.0, 1.0, 0.0, 0.0])

        new_logits = apply_top_p(logits, 0.6)
        actual_probs = mx.softmax(new_logits.squeeze())
        self.assertEqual(
            [round(p, 4) for p in actual_probs.tolist()], [0.0, 0.5556, 0.4444, 0.0]
        )

        new_logits = apply_top_p(logits, 0.95)
        actual_probs = mx.softmax(new_logits.squeeze())
        actual_rounded = [round(p, 4) for p in actual_probs.tolist()]
        expected_rounded = [0.0, 0.5, 0.4, 0.1]
        self.assertEqual(actual_rounded, expected_rounded)
        self.assertAlmostEqual(sum(actual_probs.tolist()), 1.0)

        # Batch mode works
        probs = mx.array([[0.9, 0.0, 0.0, 0.1], [0.0, 0.8, 0.1, 0.1]])
        logits = mx.log(probs)
        new_logits = apply_top_p(logits, 0.5)
        actual_probs = mx.softmax(new_logits, axis=-1)
        self.assertEqual(
            actual_probs.tolist(), [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )

    def test_apply_min_p(self):
        probs = mx.array([0.9, 0.0, 0.0, 0.1])[None]
        logits = mx.log(probs)
        new_logits = apply_min_p(logits, 0.8)
        actual_probs = mx.softmax(new_logits.squeeze())
        self.assertEqual(actual_probs.tolist(), [1.0, 0.0, 0.0, 0.0])

        probs = mx.array([0.9, 0.0, 0.0, 0.1])[None]
        logits = mx.log(probs)
        new_logits = apply_min_p(logits, 0.05)
        actual_probs = mx.softmax(new_logits.squeeze())
        self.assertTrue(mx.allclose(actual_probs, mx.squeeze(probs)))

        # Batch mode works
        probs = mx.array([[0.9, 0.0, 0.0, 0.1], [0.0, 0.8, 0.0, 0.1]])
        logits = mx.log(probs)
        new_logits = apply_min_p(logits, 0.7)
        actual_probs = mx.softmax(new_logits, axis=-1)
        self.assertEqual(
            actual_probs.tolist(), [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )

    def test_apply_top_k(self):
        probs = mx.array([0.9, 0.0, 0.0, 0.1])[None]
        logits = mx.log(probs)

        new_logits = apply_top_k(logits, 1)
        actual_probs = mx.softmax(new_logits.squeeze())
        self.assertEqual(actual_probs.tolist(), [1.0, 0.0, 0.0, 0.0])

        probs = mx.array([0.6, 0.0, 0.1, 0.3])[None]
        logits = mx.log(probs)
        new_logits = apply_top_k(logits, 2)
        actual_probs = mx.softmax(new_logits.squeeze())
        self.assertEqual(
            [round(p, 4) for p in actual_probs.tolist()], [0.6667, 0.0, 0.0, 0.3333]
        )

        # Batch mode works
        probs = mx.array([[0.9, 0.0, 0.0, 0.1], [0.0, 0.8, 0.0, 0.1]])
        logits = mx.log(probs)

        new_logits = apply_top_k(logits, 1)
        actual_probs = mx.softmax(new_logits, axis=-1)
        self.assertEqual(
            actual_probs.tolist(), [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )


if __name__ == "__main__":
    unittest.main()
