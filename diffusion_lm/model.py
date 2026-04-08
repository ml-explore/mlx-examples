# Copyright © 2024 Apple Inc.
#
# Diffusion Language Model (LLaDA-style) for MLX.
# Based on "LLaDA: Large Language Diffusion with mAsking"
# https://arxiv.org/abs/2502.09992

import inspect
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class ModelArgs:
    # Model dimensions
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 32
    mlp_hidden_size: int = 14336

    # Vocabulary
    vocab_size: int = 126464
    mask_token_id: int = 126336

    # Normalization
    rms_norm_eps: float = 1e-5

    # RoPE
    rope_theta: float = 500000.0

    @classmethod
    def from_dict(cls, params: dict) -> "ModelArgs":
        valid = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in params.items() if k in valid})


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.d_model // args.n_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(args.d_model, args.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            args.d_model, args.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.d_model, args.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(args.n_heads * self.head_dim, args.d_model, bias=False)

        self.rope = nn.RoPE(self.head_dim, traditional=False, base=args.rope_theta)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Reshape to (B, n_heads, L, head_dim)
        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        # Apply rotary embeddings
        queries = self.rope(queries)
        keys = self.rope(keys)

        # Expand KV heads for grouped query attention
        if self.n_kv_heads != self.n_heads:
            n_repeat = self.n_heads // self.n_kv_heads
            keys = mx.repeat(keys, n_repeat, axis=1)
            values = mx.repeat(values, n_repeat, axis=1)

        # Bidirectional (non-causal) scaled dot-product attention
        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores = scores + mask
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(queries.dtype)
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # SwiGLU: silu(gate_proj(x)) * up_proj(x) -> down_proj
        self.gate_proj = nn.Linear(args.d_model, args.mlp_hidden_size, bias=False)
        self.up_proj = nn.Linear(args.d_model, args.mlp_hidden_size, bias=False)
        self.down_proj = nn.Linear(args.mlp_hidden_size, args.d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.d_model, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.d_model, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        # Pre-norm attention (no causal mask — bidirectional)
        r = self.self_attn(self.input_layernorm(x), mask=mask)
        h = x + r
        # Pre-norm FFN
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Model(nn.Module):
    """Bidirectional transformer mask predictor for masked diffusion LM.

    Identical to a decoder-only LLM (LLaMA-style) except that attention is
    *non-causal* (bidirectional), allowing each position to attend to all
    other positions.  No KV-cache is used because every forward pass sees the
    full (partially masked) sequence.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = [TransformerBlock(args) for _ in range(args.n_layers)]
        self.norm = nn.RMSNorm(args.d_model, eps=args.rms_norm_eps)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            input_ids: Token ids of shape (B, L). Masked positions should
                contain ``args.mask_token_id``.
            attention_mask: Optional boolean/float padding mask of shape (B, L).
                1 = attend, 0 = ignore (padding).

        Returns:
            Logits of shape (B, L, vocab_size).
        """
        x = self.embed_tokens(input_ids)

        # Build additive padding mask: (B, 1, 1, L) → broadcast over heads/query
        mask = None
        if attention_mask is not None:
            # attention_mask: 1 where real token, 0 where padding
            # Convert to large negative for softmax
            pad = (1.0 - attention_mask.astype(mx.float32)) * -1e9
            mask = pad[:, None, None, :]  # (B, 1, 1, L)

        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.norm(x)
        return self.lm_head(x)
