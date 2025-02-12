import unittest
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import get_batched_logps, dpo_loss
from mlx_lm.tuner.trainer import train, TrainingArgs
from unittest.mock import MagicMock

class TestDPO(unittest.TestCase):

    def setUp(self):
        self.model = MagicMock()
        self.inputs = mx.array([[1, 2, 3], [4, 5, 6]])
        self.targets = mx.array([[1, 2, 3], [4, 5, 6]])
        self.reference_chosen_logps = mx.array([0.1, 0.2])
        self.reference_rejected_logps = mx.array([0.3, 0.4])
        self.beta = 0.1
        self.label_smoothing = 0.0

    def test_get_batched_logps(self):
        self.model.return_value = (mx.array([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]]), None)
        chosen_logps, rejected_logps = get_batched_logps(self.model, self.inputs, self.targets)
        np.testing.assert_array_almost_equal(chosen_logps.asnumpy(), np.array([0.1, 0.7]))
        np.testing.assert_array_almost_equal(rejected_logps.asnumpy(), np.array([0.3, 0.9]))

    def test_dpo_loss(self):
        self.model.return_value = (mx.array([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]]), None)
        loss, chosen_rewards, rejected_rewards, reward_accuracies, reward_margins, ntoks = dpo_loss(
            self.model, self.beta, self.label_smoothing, self.reference_chosen_logps, self.reference_rejected_logps, self.inputs, self.targets
        )
        self.assertAlmostEqual(loss.item(), -0.6931472)
        self.assertAlmostEqual(chosen_rewards.item(), 0.0)
        self.assertAlmostEqual(rejected_rewards.item(), 0.0)
        self.assertAlmostEqual(reward_accuracies.item(), 0.0)
        self.assertAlmostEqual(reward_margins.item(), 0.0)
        self.assertEqual(ntoks.item(), 6)

    def test_train_with_dpo_loss(self):
        train_dataset = MagicMock()
        val_dataset = MagicMock()
        tokenizer = MagicMock()
        optimizer = MagicMock()
        args = TrainingArgs(loss_type="dpo", beta=self.beta, label_smoothing=self.label_smoothing)
        train(self.model, tokenizer, optimizer, train_dataset, val_dataset, args=args)
        self.model.assert_called()

if __name__ == "__main__":
    unittest.main()
