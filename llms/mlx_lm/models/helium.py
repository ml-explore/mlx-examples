from typing import Any, Optional, Tuple
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, scaled_dot_product_attention, create_attention_mask

@dataclass
class ModelArgs(BaseModelArgs):
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    vocab_size: int
    attention_bias: bool
    attention_dropout: float
    head_dim: int
    initializer_range: float
    max_position_embeddings: int
    mlp_bias: bool
    model_type: str = "helium"
    rope_theta: float = 100000.0
    tie_word_embeddings: bool = False


def rotate_half(x: mx.array) -> mx.array:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array, position_ids=None, unsqueeze_dim=1) -> Tuple[mx.array, mx.array]:
    """
    Applies Rotary Position Embedding to the query and key tensors.
    
    Args:
        q: Query tensor
        k: Key tensor
        cos: Cosine part of the rotary embedding
        sin: Sine part of the rotary embedding
        position_ids: Deprecated and unused
        unsqueeze_dim: Dimension to unsqueeze for broadcasting
    """
    # Unsqueeze cos and sin
    for _ in range(unsqueeze_dim):
        cos = mx.expand_dims(cos, 1)
        sin = mx.expand_dims(sin, 1)
    
    # Interleave the cos and sin values
    cos = mx.repeat(cos[..., :cos.shape[-1] // 2], repeats=2, axis=-1)
    sin = mx.repeat(sin[..., :sin.shape[-1] // 2], repeats=2, axis=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


def apply_rotary_pos_emb(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array, position_ids=None, unsqueeze_dim=1) -> Tuple[mx.array, mx.array]:
    """
    Applies Rotary Position Embedding to the query and key tensors.
    
    Args:
        q: Query tensor (batch, n_heads, seq_len, head_dim)
        k: Key tensor (batch, n_heads, seq_len, head_dim)
        cos: Cosine part of rotary embedding (batch, seq_len, head_dim)
        sin: Sine part of rotary embedding (batch, seq_len, head_dim)
    """
    # Reshape cos and sin to match the query/key shape
    cos = mx.expand_dims(cos, axis=1)  # (batch, 1, seq_len, head_dim)
    sin = mx.expand_dims(sin, axis=1)  # (batch, 1, seq_len, head_dim)
    
    # Make sure we only rotate half of the dimensions
    head_dim = q.shape[-1]
    cos = mx.repeat(cos[..., :head_dim//2], repeats=2, axis=-1)
    sin = mx.repeat(sin[..., :head_dim//2], repeats=2, axis=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class HeliumRotaryEmbedding(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.base = config.rope_theta

    def __call__(self, x: mx.array, position_ids: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Args:
            x: Input tensor (batch, seq_len, hidden_size)
            position_ids: Position IDs (batch, seq_len)
        Returns:
            Tuple of (cos, sin) tensors for rotary embeddings
        """
        batch_size, seq_length = position_ids.shape
        
        # Initialize output tensors for cos and sin
        cos_cached = []
        sin_cached = []
        
        # Generate embeddings for each position
        for i in range(seq_length):
            # Create position-specific embedding
            theta = 1.0 / (self.base ** (mx.arange(self.head_dim//2) / (self.head_dim//2)))
            pos_embedding = i * theta
            
            # Calculate cos and sin
            cos = mx.cos(pos_embedding)
            sin = mx.sin(pos_embedding)
            
            cos_cached.append(cos)
            sin_cached.append(sin)
        
        # Stack along sequence dimension
        cos_cached = mx.stack(cos_cached, axis=0)  # (seq_len, head_dim//2)
        sin_cached = mx.stack(sin_cached, axis=0)  # (seq_len, head_dim//2)
        
        # Add batch dimension and expand
        cos_cached = mx.expand_dims(cos_cached, axis=0)  # (1, seq_len, head_dim//2)
        sin_cached = mx.expand_dims(sin_cached, axis=0)  # (1, seq_len, head_dim//2)
        
        # Repeat for batch size
        cos_cached = mx.repeat(cos_cached, batch_size, axis=0)  # (batch, seq_len, head_dim//2)
        sin_cached = mx.repeat(sin_cached, batch_size, axis=0)  # (batch, seq_len, head_dim//2)
        
        return cos_cached, sin_cached


class HeliumAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        assert args.num_key_value_heads is not None
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        head_dim = args.hidden_size // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

    def __call__(
        self,
        x: mx.array,
        position_embeddings: tuple[mx.array, mx.array],  # (cos, sin)
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # Apply rotary embeddings
        cos, sin = position_embeddings
        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class HeliumMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.intermediate_size = args.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=args.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=args.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=args.mlp_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class HeliumDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size

        self.self_attn = HeliumAttention(args)
        self.mlp = HeliumMLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
    
    def __call__(
        self,
        x: mx.array,
        position_embeddings: tuple[mx.array, mx.array],
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), position_embeddings, mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class HeliumModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_hidden_layers = args.num_hidden_layers
        self.vocab_size = args.vocab_size

        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)

        self.layers = [
            HeliumDecoderLayer(args) for _ in range(args.num_hidden_layers)
        ]

        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        
        # Create RoPE embeddings to be shared across layers
        self.rotary_emb = HeliumRotaryEmbedding(args)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)

        if mask is None:
            mask = create_attention_mask(h, cache)

        # Generate position embeddings once to be shared across layers
        position_embeddings = self.rotary_emb(h, inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, position_embeddings, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type

        self.model = HeliumModel(args)

        self.vocab_size = args.vocab_size
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ) -> mx.array:
        out = self.model(inputs, mask, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out
    
    @property
    def layers(self):
        return self.model.layers