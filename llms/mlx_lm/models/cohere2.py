from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope
from .cache import KVCache, RotatingKVCache

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    rope_theta: float
    vocab_size: int
    layer_norm_eps: float
    logit_scale: float
    attention_bias: bool
    # Additional Cohere2-specific arguments:
    # rope_type and max_position_embeddings might influence the rope setup
    rope_type: str = "default"
    max_position_embeddings: int = 2048
    sliding_window: Optional[int] = None,
    sliding_window_pattern: Optional[int] = None,
    order_of_interleaved_layers: Optional[int] = None,
    use_cache: bool = True



class Cohere2Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        head_dim = dim // self.n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * head_dim, dim, bias=args.attention_bias)

        self.sliding_window = args.sliding_window # Not yet implemented :(
        self.use_qk_norm = False  # Assuming QK norm not used by Cohere2 (adjust if needed)

        # Initialize RoPE for Cohere2
        self.rope = initialize_rope(
            dims=head_dim,
            base=args.rope_theta,
            traditional=True,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None, rope = True) -> mx.array:
        B, L, D = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        # Apply RoPE
        # In Cohere2, the original code applies RoPE before caching updates. We replicate that:
        if cache is not None:
            if rope:
                q = self.rope(q, offset=cache.offset)
                k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
            if rope:
                k = k[:, :, -self.sliding_window:, :]
                v = v[:, :, -self.sliding_window:, :]
        elif rope:
            q = self.rope(q)
            k = self.rope(k)
        # Compute attention
        out = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )

        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.o_proj(out)


class Cohere2MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        hdim = args.intermediate_size
        self.gate_proj = nn.Linear(dim, hdim, bias=False)
        self.up_proj = nn.Linear(dim, hdim, bias=False)
        self.down_proj = nn.Linear(hdim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Cohere2TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Cohere2Attention(args)
        self.mlp = Cohere2MLP(args)
        self.input_layernorm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps, affine=True, bias=False)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None, rope = True) -> mx.array:
        h = self.input_layernorm(x)
        attn_h = self.self_attn(h, mask, cache, rope=rope)
        ff_h = self.mlp(h)
        return x + attn_h + ff_h


class Cohere2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [Cohere2TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps, affine=True, bias=False)
        self.sliding_window = args.sliding_window
        self.sliding_window_pattern = args.sliding_window_pattern
    def __call__(self, inputs: mx.array, cache: Optional[Any] = None) -> mx.array:
        h = self.embed_tokens(inputs)
        mask = create_attention_mask(h, cache, reference_cache_idx=self.sliding_window_pattern - 1)
        sliding_window_mask = mask[:, -self.sliding_window:] if mask is not None else None
        if cache is None:
            cache = [None] * len(self.layers)
        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            if self.sliding_window is not None:
                index = i % self.sliding_window_pattern
                if index < self.sliding_window_pattern - 1:
                    h = layer(h, mask=sliding_window_mask, cache=c)
                else:
                    h = layer(h, mask=mask, cache=c, rope=False)


        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type

        self.model = Cohere2Model(args)
        self.args = args

    def __call__(self, inputs: mx.array, cache=None):
        out = self.model(inputs, cache)
        out = self.model.embed_tokens.as_linear(out) * self.args.logit_scale
        return out

    @property
    def layers(self):
        return self.model.layers
    
    def make_cache(self):
        caches = []
        for i in range(self.args.num_hidden_layers):
            if i % self.args.sliding_window_pattern == self.args.sliding_window_pattern - 1:
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window, keep=0))
        return caches
