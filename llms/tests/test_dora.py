import math
import sys
import unittest
from io import StringIO

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import tuner
from mlx_lm.tuner.dora import DoRAEmbedding


class TestDoRAEmbedding(unittest.TestCase):
    def test_dora_embedding(self):
        num_embeddings = 256
        dims = 512
        tokens = mx.array([1, 2, 3])

        embedding = nn.Embedding(num_embeddings, dims)

        dora_emb = DoRAEmbedding.from_embedding(embedding, r=8, dropout=0, scale=10)
        new_embedding = dora_emb.to_embedding()
        self.assertTrue(mx.array_equal(embedding.weight, new_embedding.weight))
        self.assertTrue(mx.array_equal(embedding(tokens), dora_emb(tokens)))

        # works with bare ints like embeddings do
        self.assertTrue(mx.array_equal(dora_emb(1), dora_emb([1])[0]))

        # as_linear
        attn_output = mx.random.uniform(shape=(dims,))
        embedding_lin_out = dora_emb.as_linear(attn_output)
        self.assertEqual(embedding_lin_out.shape, (num_embeddings,))
        self.assertTrue(
            mx.array_equal(embedding_lin_out, embedding.as_linear(attn_output))
        )

        # change the value of lora_b and the embeddings will no longer be equal
        dora_emb.lora_b = mx.random.uniform(shape=dora_emb.lora_b.shape)
        new_embedding = dora_emb.to_embedding()
        self.assertFalse(mx.array_equal(embedding.weight, new_embedding.weight))
        self.assertFalse(mx.array_equal(embedding(tokens), dora_emb(tokens)))
