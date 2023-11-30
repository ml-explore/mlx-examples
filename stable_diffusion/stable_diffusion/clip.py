# Copyright © 2023 Apple Inc.

import mlx.core as mx
import mlx.nn as nn

from .config import CLIPTextModelConfig


class CLIPEncoderLayer(nn.Module):
    """The transformer encoder layer from CLIP."""

    def __init__(self, model_dims: int, num_heads: int):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(model_dims)
        self.layer_norm2 = nn.LayerNorm(model_dims)

        self.attention = nn.MultiHeadAttention(model_dims, num_heads)
        # Add biases to the attention projections to match CLIP
        self.attention.query_proj.bias = mx.zeros(model_dims)
        self.attention.key_proj.bias = mx.zeros(model_dims)
        self.attention.value_proj.bias = mx.zeros(model_dims)
        self.attention.out_proj.bias = mx.zeros(model_dims)

        self.linear1 = nn.Linear(model_dims, 4 * model_dims)
        self.linear2 = nn.Linear(4 * model_dims, model_dims)

    def __call__(self, x, attn_mask=None):
        y = self.layer_norm1(x)
        y = self.attention(y, y, y, attn_mask)
        x = y + x

        y = self.layer_norm2(x)
        y = self.linear1(y)
        y = nn.gelu_approx(y)
        y = self.linear2(y)
        x = y + x

        return x


class CLIPTextModel(nn.Module):
    """Implements the text encoder transformer from CLIP."""

    def __init__(self, config: CLIPTextModelConfig):
        super().__init__()

        self.token_embedding = nn.Embedding(config.vocab_size, config.model_dims)
        self.position_embedding = nn.Embedding(config.max_length, config.model_dims)
        self.layers = [
            CLIPEncoderLayer(config.model_dims, config.num_heads)
            for i in range(config.num_layers)
        ]
        self.final_layer_norm = nn.LayerNorm(config.model_dims)

    def __call__(self, x):
        # Extract some shapes
        B, N = x.shape

        # Compute the embeddings
        x = self.token_embedding(x)
        x = x + self.position_embedding.weight[:N]

        # Compute the features from the transformer
        mask = nn.MultiHeadAttention.create_additive_causal_mask(N, x.dtype)
        for l in self.layers:
            x = l(x, mask)

        # Apply the final layernorm and return
        return self.final_layer_norm(x)
