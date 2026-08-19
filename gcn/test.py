import unittest

import mlx.core as mx
import mlx.nn as nn
from main import evaluate


class ModeSensitiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_modes = []

    def __call__(self, x, adj):
        self.forward_modes.append(self.training)
        if self.training:
            return mx.array([[0.0, 1.0], [1.0, 0.0]])
        return mx.array([[1.0, 0.0], [0.0, 1.0]])


class TestEvaluation(unittest.TestCase):
    def test_evaluate_uses_eval_mode_and_restores_model_state(self):
        labels = mx.array([0, 1])
        mask = mx.arange(2)

        for training in (True, False):
            with self.subTest(training=training):
                model = ModeSensitiveModel().train(training)

                loss, accuracy = evaluate(model, None, None, labels, mask)

                self.assertEqual(model.forward_modes, [False])
                self.assertEqual(model.training, training)
                self.assertAlmostEqual(accuracy.item(), 1.0)
                self.assertAlmostEqual(loss.item(), 0.31326166, places=6)


if __name__ == "__main__":
    unittest.main()
