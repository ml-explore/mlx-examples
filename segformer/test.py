import unittest
from models import (
    SegformerLayer,
    SegformerEfficientSelfAttention,
    SegformerOverlapPatchEmbeddings,
)
import numpy as np
import mlx.core as mx


class TestSegformer(unittest.TestCase):
    def test_overlap_patch_embeddings(self):
        input_tensor = mx.random.uniform(shape=(1, 224, 224, 3))
        overlap_patch_embeddings = SegformerOverlapPatchEmbeddings(
            patch_size=7,
            stride=3,
            num_channels=3,
            hidden_size=32,
        )
        output = overlap_patch_embeddings.forward(input_tensor)
        self.assertEqual(output.shape, (1, 75, 75, 32))

    def test_segformer_efficient_self_attention(self):
        hidden_size = 160
        num_attention_heads = 5
        input_tensor = mx.random.uniform(shape=(1, 25, 25, hidden_size))
        efficient_self_attention = SegformerEfficientSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=2,
        )
        output = efficient_self_attention.forward(input_tensor)
        self.assertEqual(
            output.shape,
            (1, 25 * 25 * num_attention_heads, hidden_size / num_attention_heads),
        )

    def test_segformer_layer(self):
        input_tensor = mx.random.uniform(shape=(1, 3, 224, 224))
        segformer_layer = SegformerLayer(
            hidden_size=160,
            num_attention_heads=5,
            sequence_reduction_ratio=2,
            mlp_ratio=4,
        )
        output = segformer_layer.forward(input_tensor)
        self.assertEqual(output.shape, (1, 25 * 25 * 5, 160 / 5))


if __name__ == "__main__":
    unittest.main()
