"""
DeciLM model implementation for MLX.
Supports Neural Architecture Search (NAS) optimized models with:
- Dummy layers (no-op attention/FFN)
- Variable Grouped Query Attention
- FFN Fusion
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import BaseModelArgs


@dataclass
class DeciLMArgs(BaseModelArgs):
    """Arguments for DeciLM model."""
    model_type: str = "decilm"
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-6
    vocab_size: int = 32000
    attention_bias: bool = False
    rope_theta: float = 10000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Any]] = None
    
    # DeciLM specific
    block_configs: Optional[list] = None  # Per-layer configurations
    
    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


class DummyAttention(nn.Module):
    """Dummy attention layer that passes input through unchanged."""
    def __init__(self, args: DeciLMArgs):
        super().__init__()
        # No parameters - just pass through
        
    def __call__(self, x, mask=None, cache=None):
        # Return input unchanged
        return x


class DummyFFN(nn.Module):
    """Dummy FFN layer that passes input through unchanged."""
    def __init__(self, args: DeciLMArgs):
        super().__init__()
        # No parameters - just pass through
        
    def __call__(self, x):
        # Return input unchanged
        return x


class VariableAttention(nn.Module):
    """Attention with variable number of KV heads per layer."""
    def __init__(self, args: DeciLMArgs, n_kv_heads: int):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads  # Variable per layer
        self.head_dim = args.hidden_size // args.num_attention_heads
        
        self.scale = self.head_dim**-0.5
        
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(args.hidden_size, self.n_kv_heads * self.head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(args.hidden_size, self.n_kv_heads * self.head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=args.attention_bias)
        
        rope_scale = 1.0
        if args.rope_scaling:
            rope_scale = args.rope_scaling.get("factor", 1.0)
            
        self.rope = nn.RoPE(self.head_dim, traditional=args.rope_traditional, base=args.rope_theta, scale=rope_scale)
        
    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        
        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Apply RoPE
        queries = self.rope(queries, offset=cache.offset if cache else 0)
        keys = self.rope(keys, offset=cache.offset if cache else 0)
        
        # Update cache if provided
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
            
        # Repeat KV heads if needed
        if self.n_kv_heads != self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            keys = mx.repeat(keys, n_rep, axis=1)
            values = mx.repeat(values, n_rep, axis=1)
            
        # Compute attention
        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale
        
        if mask is not None:
            scores = scores + mask
            
        scores = mx.softmax(scores, axis=-1)
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        
        return self.o_proj(output)


class VariableFFN(nn.Module):
    """FFN with variable expansion ratio."""
    def __init__(self, args: DeciLMArgs, ffn_mult: float):
        super().__init__()
        # Calculate intermediate size based on multiplier
        intermediate_size = int(args.hidden_size * ffn_mult)
        
        self.gate_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, args.hidden_size, bias=False)
        
    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DeciLMBlock(nn.Module):
    """Transformer block with DeciLM variable architecture."""
    def __init__(self, args: DeciLMArgs, block_config: dict):
        super().__init__()
        self.args = args
        self.block_config = block_config
        
        # Layer norms always present
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        
        # Attention layer (can be dummy)
        attn_config = block_config["attention"]
        if attn_config.get("no_op", False):
            self.self_attn = DummyAttention(args)
        else:
            n_kv_heads = attn_config.get("n_heads_in_group", args.num_key_value_heads)
            self.self_attn = VariableAttention(args, n_kv_heads)
            
        # FFN layer (can be dummy)
        ffn_config = block_config["ffn"]
        if ffn_config.get("no_op", False):
            self.mlp = DummyFFN(args)
        else:
            ffn_mult = ffn_config.get("ffn_mult", 2.5)
            self.mlp = VariableFFN(args, ffn_mult)
            
    def __call__(self, x, mask=None, cache=None):
        # Self attention (may be dummy/no-op)
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        
        # FFN (may be dummy/no-op)
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        
        return out


class DeciLMModel(nn.Module):
    """DeciLM model with NAS-optimized architecture."""
    def __init__(self, args: DeciLMArgs):
        super().__init__()
        self.args = args
        
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        
        # Build layers with per-layer configs
        self.layers = []
        for i, block_config in enumerate(args.block_configs):
            self.layers.append(DeciLMBlock(args, block_config))
            
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        
    def __call__(self, inputs, cache=None):
        h = self.embed_tokens(inputs)
        
        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)
            
        if cache is None:
            cache = [None] * len(self.layers)
            
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
            
        return self.norm(h)


class Model(nn.Module):
    """Full DeciLM model for generation."""
    def __init__(self, args: DeciLMArgs):
        super().__init__()
        self.args = args
        self.model = DeciLMModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        
    def __call__(self, inputs, cache=None):
        out = self.model(inputs, cache)
        return self.lm_head(out)
        
    def sanitize(self, weights):
        # Convert weights if needed
        return weights
        
    @property
    def layers(self):
        return self.model.layers