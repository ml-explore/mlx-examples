# Copyright © 2023 Apple Inc.

import mlx.core as mx
import mlx.nn as nn
from config import CLIPTextConfig, CLIPVisionConfig


def quick_gelu(x: mx.array) -> mx.array:
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """
    return x * mx.sigmoid(1.702 * x)


class CLIPEncoderLayer(nn.TransformerEncoderLayer):
    """The transformer encoder layer from CLIP."""

    def __init__(self, hidden_dim: int, intermediate_dim: int, num_heads: int):
        super().__init__(
            dims=hidden_dim,
            mlp_dims=intermediate_dim,
            num_heads=num_heads,
            activation=quick_gelu,
            norm_first=True
        )
        self.attention.query_proj.bias = mx.zeros(hidden_dim)
        self.attention.key_proj.bias = mx.zeros(hidden_dim)
        self.attention.value_proj.bias = mx.zeros(hidden_dim)
        self.attention.out_proj.bias = mx.zeros(hidden_dim)


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
