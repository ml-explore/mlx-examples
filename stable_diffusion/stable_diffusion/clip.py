# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from .config import CLIPTextModelConfig

_ACTIVATIONS = {"quick_gelu": nn.gelu_fast_approx, "gelu": nn.gelu}


@dataclass
class CLIPOutput:
    # The last_hidden_state indexed at the EOS token and possibly projected if
    # the model has a projection layer
    pooled_output: Optional[mx.array] = None

    # The full sequence output of the transformer after the final layernorm
    last_hidden_state: Optional[mx.array] = None

    # A list of hidden states corresponding to the outputs of the transformer layers
    hidden_states: Optional[List[mx.array]] = None


class CLIPEncoderLayer(nn.Module):
    """The transformer encoder layer from CLIP."""

    def __init__(self, model_dims: int, num_heads: int, activation: str):
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

        self.act = _ACTIVATIONS[activation]

    def __call__(self, x, attn_mask=None):
        y = self.layer_norm1(x)
        y = self.attention(y, y, y, attn_mask)
        x = y + x

        y = self.layer_norm2(x)
        y = self.linear1(y)
        y = self.act(y)
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
            CLIPEncoderLayer(config.model_dims, config.num_heads, config.hidden_act)
            for i in range(config.num_layers)
        ]
        self.final_layer_norm = nn.LayerNorm(config.model_dims)

        if config.projection_dim is not None:
            self.text_projection = nn.Linear(
                config.model_dims, config.projection_dim, bias=False
            )

    def _get_mask(self, N, dtype):
        indices = mx.arange(N)
        mask = indices[:, None] < indices[None]
        mask = mask.astype(dtype) * (-6e4 if dtype == mx.float16 else -1e9)
        return mask

    def __call__(self, x):
        # Extract some shapes
        B, N = x.shape
        eos_tokens = x.argmax(-1)

        # Compute the embeddings
        x = self.token_embedding(x)
        x = x + self.position_embedding.weight[:N]

        # Compute the features from the transformer
        mask = self._get_mask(N, x.dtype)
        hidden_states = []
        for l in self.layers:
            x = l(x, mask)
            hidden_states.append(x)

        # Apply the final layernorm and return
        x = self.final_layer_norm(x)
        last_hidden_state = x

        # Select the EOS token
        pooled_output = x[mx.arange(len(x)), eos_tokens]
        if "text_projection" in self:
            pooled_output = self.text_projection(pooled_output)

        return CLIPOutput(
            pooled_output=pooled_output,
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states,
        )
