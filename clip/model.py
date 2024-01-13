# Copyright © 2023 Apple Inc.

import mlx.core as mx
import mlx.nn as nn
from config import CLIPTextConfig, CLIPVisionConfig


def quick_gelu(x: mx.array) -> mx.array:
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """
    return x * mx.sigmoid(1.702 * x)


class CLIPEncoderLayer(nn.Module):
    """The transformer encoder layer from CLIP."""

    def __init__(self, hidden_dim: int, intermediate_dim: int, num_heads: int):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        self.attention = nn.MultiHeadAttention(hidden_dim, num_heads)
        # Add biases to the attention projections to match CLIP
        self.attention.query_proj.bias = mx.zeros(hidden_dim)
        self.attention.key_proj.bias = mx.zeros(hidden_dim)
        self.attention.value_proj.bias = mx.zeros(hidden_dim)
        self.attention.out_proj.bias = mx.zeros(hidden_dim)

        self.linear1 = nn.Linear(hidden_dim, intermediate_dim)
        self.linear2 = nn.Linear(intermediate_dim, hidden_dim)

    def __call__(self, x, attn_mask=None):
        y = self.layer_norm1(x)
        y = self.attention(y, y, y, attn_mask)
        x = y + x

        y = self.layer_norm2(x)
        y = self.linear1(y)
        y = quick_gelu(y)
        y = self.linear2(y)
        x = y + x

        return x


class CLIPTextModel(nn.Module):
    """Implements the text encoder transformer from CLIP."""

    def __init__(self, config: CLIPTextConfig):
        super().__init__()

        self.token_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.layers = [
            CLIPEncoderLayer(
                config.hidden_size,
                config.intermediate_size,
                config.num_attention_heads
            )
            for _ in range(config.num_hidden_layers)
        ]
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

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


class CLIPVisionModel(nn.Module):
    """Implements the vision encoder transformer from CLIP."""

    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        pass

    def __call__(self, x):
        pass
