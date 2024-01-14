# Copyright © 2023 Apple Inc.

from dataclasses import dataclass
from functools import reduce

import mlx.core as mx
import mlx.nn as nn
from config import CLIPConfig, CLIPTextConfig, CLIPVisionConfig


def quick_gelu(x: mx.array) -> mx.array:
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """
    return x * mx.sigmoid(1.702 * x)


@dataclass
class CLIPVisionOutput:
    pooler_output: mx.array
    last_hidden_state: mx.array


@dataclass
class CLIPTextOutput:
    pooler_output: mx.array
    last_hidden_state: mx.array


class CLIPEncoderLayer(nn.TransformerEncoderLayer):
    """The transformer encoder layer from CLIP."""

    def __init__(self, hidden_dim: int, intermediate_dim: int, num_heads: int):
        super().__init__(
            dims=hidden_dim,
            mlp_dims=intermediate_dim,
            num_heads=num_heads,
            activation=quick_gelu,
            norm_first=True,
        )
        self.attention.query_proj.bias = mx.zeros(hidden_dim)
        self.attention.key_proj.bias = mx.zeros(hidden_dim)
        self.attention.value_proj.bias = mx.zeros(hidden_dim)
        self.attention.out_proj.bias = mx.zeros(hidden_dim)


class CLIPTextModel(nn.Module):
    """Implements the text encoder transformer from CLIP."""

    def __init__(self, config: CLIPTextConfig):
        super().__init__()

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = mx.zeros(
            (config.max_position_embeddings, config.hidden_size)
        )
        self.layers = [
            CLIPEncoderLayer(
                config.hidden_size, config.intermediate_size, config.num_attention_heads
            )
            for _ in range(config.num_hidden_layers)
        ]
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

    def _embed(self, x: mx.array) -> mx.array:
        # Extract some shapes
        _, N = x.shape
        # Compute the embeddings
        embeddings = self.token_embedding(x)
        embeddings += self.position_embedding[:N]
        return embeddings

    def __call__(self, x: mx.array) -> CLIPTextOutput:
        # Extract some shapes
        B, N = x.shape
        eot_tokens = mx.argmax(x, axis=-1)
        # Look up embeddings
        x = self._embed(x)
        # Compute the causal mask
        mask = nn.MultiHeadAttention.create_additive_causal_mask(N, x.dtype)
        # Push through the transformer
        x = reduce(lambda x, l: l(x, mask), self.layers, x)
        # Apply the final layernorm
        last_hidden_state = self.final_layer_norm(x)
        pooler_output = last_hidden_state[mx.arange(B), eot_tokens]

        return CLIPTextOutput(
            pooler_output=pooler_output, last_hidden_state=last_hidden_state
        )


class CLIPVisionModel(nn.Module):
    """Implements the vision encoder transformer from CLIP."""

    def __init__(self, config: CLIPVisionConfig):
        super().__init__()

        self.class_embedding = mx.random.normal((config.hidden_size,))
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        num_patches = (config.image_size // config.patch_size) ** 2
        num_positions = num_patches + 1
        self.position_embedding = mx.random.normal((num_positions, config.hidden_size))
        self.pre_layernorm = nn.LayerNorm(config.hidden_size)
        self.layers = [
            CLIPEncoderLayer(
                config.hidden_size, config.intermediate_size, config.num_attention_heads
            )
            for _ in range(config.num_hidden_layers)
        ]
        self.post_layernorm = nn.LayerNorm(config.hidden_size)

    def _embed(self, x: mx.array) -> mx.array:
        [batch_size, _, _, _] = x.shape
        # Patchify using conv; [batch_size, sqrt(num_patches), sqrt(num_patches), embed_dim]
        patch_embeddings = self.patch_embedding(x)
        # [batch_size, num_patches, embed_dim]
        patch_embeddings = mx.flatten(patch_embeddings, start_axis=1, end_axis=2)
        [_, _, embed_dim] = patch_embeddings.shape
        # Append <CLS> embeddings
        # [batch_size, 1, embed_dim]
        cls_embeddings = mx.broadcast_to(
            self.class_embedding, (batch_size, 1, embed_dim)
        )
        # [batch_size, num_patches + 1, embed_dim]
        embeddings = mx.concatenate((cls_embeddings, patch_embeddings), axis=1)
        # Add positional encoding
        embeddings += self.position_embedding
        return embeddings

    def __call__(self, x: mx.array) -> CLIPVisionOutput:
        # Look up patch embeddings
        x = self._embed(x)
        # Prenorm
        x = self.pre_layernorm(x)
        # Push through transformer
        x = reduce(lambda x, l: l(x, mask=None), self.layers, x)
        # Pool <CLS> token
        pooler_output = self.post_layernorm(x[:, 0, :])
        return CLIPVisionOutput(pooler_output=pooler_output, last_hidden_state=x)
