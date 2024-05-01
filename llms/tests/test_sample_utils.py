import unittest
from unittest.mock import patch

import mlx.core as mx
from mlx_lm.sample_utils import top_p_sampling


class TestSamplingUtils(unittest.TestCase):
    @patch("mlx.core.random.categorical")
    def test_top_p_sampling(self, mock_categorical):
        logits = mx.array([[1.0, 2.0, 3.0, 4.0]])
        top_p = 0.3
        temperature = 1.0
        expected_token = mx.array([3])
        mock_categorical.return_value = expected_token

        token = top_p_sampling(logits, top_p, temperature)
        expected_top_probs = mx.array([[0.0, 0.0, 0.0, 0.643914]])
        self.assertTrue(mx.allclose(token, expected_token))
        args, _ = mock_categorical.call_args
        self.assertTrue(args[0].shape == expected_top_probs.shape)
        self.assertTrue(mx.allclose(args[0], mx.log(expected_top_probs)))

        logits = mx.array([[1.0, 2.0, 3.0, 4.0]])
        top_p = 0.9
        temperature = 1.0
        expected_token = mx.array([3])
        mock_categorical.return_value = expected_token

        token = top_p_sampling(logits, top_p, temperature)
        expected_top_probs = mx.array([[0.0, 0.0871443, 0.236883, 0.643914]])
        self.assertTrue(mx.allclose(token, expected_token))
        args, _ = mock_categorical.call_args
        self.assertTrue(args[0].shape == expected_top_probs.shape)
        self.assertTrue(mx.allclose(args[0], mx.log(expected_top_probs)))


if __name__ == "__main__":
    unittest.main()
