import unittest
import numpy as np
import mlx.nn as nn
from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten
from transformers import PreTrainedTokenizerFast
from llms.mlx_lm.tuner.trainer import default_loss, instruct_loss

class TestLossFunctions(unittest.TestCase):

    def setUp(self):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("gpt2")
        self.model = nn.Module()
        self.inputs = np.array([[1, 2, 3], [4, 5, 6]])
        self.targets = np.array([[1, 2, 3], [4, 5, 6]])
        self.lengths = np.array([3, 3])

    def test_default_loss(self):
        loss, ntoks = default_loss(self.model, self.inputs, self.targets, self.lengths)
        self.assertIsInstance(loss, nn.Tensor)
        self.assertIsInstance(ntoks, nn.Tensor)

    def test_instruct_loss(self):
        loss, ntoks = instruct_loss(self.model, self.inputs, self.targets, self.lengths)
        self.assertIsInstance(loss, nn.Tensor)
        self.assertIsInstance(ntoks, nn.Tensor)

    def test_instruct_loss_with_masking(self):
        loss, ntoks = instruct_loss(self.model, self.inputs, self.targets, self.lengths, mask_input=True)
        self.assertIsInstance(loss, nn.Tensor)
        self.assertIsInstance(ntoks, nn.Tensor)

    def test_instruct_loss_without_masking(self):
        loss, ntoks = instruct_loss(self.model, self.inputs, self.targets, self.lengths, mask_input=False)
        self.assertIsInstance(loss, nn.Tensor)
        self.assertIsInstance(ntoks, nn.Tensor)

if __name__ == "__main__":
    unittest.main()
