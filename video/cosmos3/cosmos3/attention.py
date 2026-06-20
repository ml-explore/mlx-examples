"""Cosmos 3 dual-pathway Mixture-of-Transformers attention.

The MoT attention has two pathways:
  - Understanding (reasoner): causal self-attention with standard Q/K/V
  - Generation (diffuser): full attention over [und + gen] tokens with separate Q/K/V

Both share the same RoPE and layer structure.
"""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .rope import Cosmos3RotaryEmbedding, apply_rotary_pos_emb


class Cosmos3Attention(nn.Module):
    """Dual-pathway packed attention for Cosmos 3 MoT."""

    def __init__(
        self,
        hidden_size: int = 4096,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        mrope_section: list[int] | None = None,
        rope_theta: float = 5_000_000.0,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # Understanding pathway Q/K/V/O projections
        self.to_q = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.to_k = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.to_v = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.to_out = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        # QK norms (per-head RMSNorm)
        self.norm_q = nn.RMSNorm(head_dim, eps=rms_norm_eps)
        self.norm_k = nn.RMSNorm(head_dim, eps=rms_norm_eps)

        # Generation pathway projections
        self.add_q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.add_k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.add_v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.to_add_out = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)
        self.norm_added_q = nn.RMSNorm(head_dim, eps=rms_norm_eps)
        self.norm_added_k = nn.RMSNorm(head_dim, eps=rms_norm_eps)

        # RoPE
        self.rope = Cosmos3RotaryEmbedding(
            head_dim=head_dim,
            mrope_section=mrope_section or [24, 20, 20],
            rope_theta=rope_theta,
        )

    def _project_and_reshape(
        self,
        x: mx.array,
        proj: nn.Linear,
        num_heads: int,
    ) -> mx.array:
        """Project and reshape to [batch, seq_len, num_heads, head_dim]."""
        batch, seq_len, _ = x.shape
        out = proj(x)
        return out.reshape(batch, seq_len, num_heads, self.head_dim)

    def __call__(
        self,
        hidden_states: mx.array,
        position_ids: mx.array,
        understanding_mask: Optional[mx.array] = None,
        generation_tokens: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[mx.array], Optional[Tuple[mx.array, mx.array]]]:
        """Forward pass.

        Args:
            hidden_states: [batch, und_len, hidden_size] understanding tokens
            position_ids: [3, batch, total_len] position IDs per axis
            understanding_mask: optional attention mask
            generation_tokens: [batch, gen_len, hidden_size] or None
            cache: optional (keys, values) KV cache tuple

        Returns:
            (und_output, gen_output, updated_cache)
            gen_output is None if generation_tokens is None
        """
        batch, und_len, _ = hidden_states.shape

        # Understanding pathway Q/K/V
        q = self._project_and_reshape(hidden_states, self.to_q, self.num_heads)
        k = self._project_and_reshape(hidden_states, self.to_k, self.num_kv_heads)
        v = self._project_and_reshape(hidden_states, self.to_v, self.num_kv_heads)

        # QK normalization
        q = self.norm_q(q)
        k = self.norm_k(k)

        # Apply RoPE
        und_position_ids = position_ids[:, :, :und_len]
        cos, sin = self.rope(und_position_ids, seq_len=und_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV cache
        if cache is not None:
            k_cache, v_cache = cache
            k = mx.concatenate([k_cache, k], axis=1)
            v = mx.concatenate([v_cache, v], axis=1)

        new_cache = (k, v)

        # Save un-expanded keys/values for generation pathway
        k_unexpanded, v_unexpanded = k, v

        # GQA: repeat KV heads to match query heads for understanding attention
        k_attn, v_attn = k, v
        if self.num_kv_heads != self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k_attn = mx.repeat(k, repeat_factor, axis=2)
            v_attn = mx.repeat(v, repeat_factor, axis=2)

        # Compute attention: [batch, seq_len, num_heads, head_dim]
        # Transpose to [batch, num_heads, seq_len, head_dim] for SDPA
        q_t = mx.transpose(q, (0, 2, 1, 3))
        k_t = mx.transpose(k_attn, (0, 2, 1, 3))
        v_t = mx.transpose(v_attn, (0, 2, 1, 3))

        # Causal attention for understanding pathway
        q_len = q_t.shape[2]
        k_len = k_t.shape[2]

        if q_len == 1:
            # Single-token generation: no mask needed
            attn_out = mx.fast.scaled_dot_product_attention(
                q_t, k_t, v_t, scale=self.scale
            )
        else:
            # Prefill or multi-token: apply causal mask
            # Create full causal mask over key length, take last q_len rows
            full_mask = nn.MultiHeadAttention.create_additive_causal_mask(
                k_len, dtype=q_t.dtype
            )
            # When cache is present, we only have q_len query positions
            # corresponding to the last q_len rows of the full causal mask
            mask = full_mask[-q_len:]
            attn_out = mx.fast.scaled_dot_product_attention(
                q_t, k_t, v_t, scale=self.scale, mask=mask
            )

        # Transpose back and project: [batch, seq_len, num_heads * head_dim]
        attn_out = mx.transpose(attn_out, (0, 2, 1, 3))
        attn_out = attn_out.reshape(batch, -1, self.num_heads * self.head_dim)
        und_output = self.to_out(attn_out)

        # Generation pathway
        gen_output = None
        if generation_tokens is not None:
            gen_output = self._generation_forward(
                generation_tokens, hidden_states,
                k_unexpanded, v_unexpanded,
                position_ids, und_len,
            )

        return und_output, gen_output, new_cache, (k_unexpanded, v_unexpanded)

    def generation_only_forward(
        self,
        gen_tokens: mx.array,
        und_kv: Tuple[mx.array, mx.array],
        position_ids: mx.array,
        und_len: int,
    ) -> mx.array:
        """Generation pathway only, using cached understanding K/V.

        Skips the entire understanding pathway (Q/K/V projection, attention, output).
        Uses pre-computed understanding keys/values for cross-attention.
        """
        return self._generation_forward(
            gen_tokens, None, und_kv[0], und_kv[1],
            position_ids, und_len,
        )

    def _generation_forward(
        self,
        gen_tokens: mx.array,
        und_tokens: mx.array,
        und_keys: mx.array,
        und_values: mx.array,
        position_ids: mx.array,
        und_len: int,
    ) -> mx.array:
        """Generation pathway: full attention over [und + gen] tokens.

        Bidirectional attention for diffusion.
        """
        batch, gen_len, _ = gen_tokens.shape

        # Generation Q/K/V
        q_gen = self._project_and_reshape(gen_tokens, self.add_q_proj, self.num_heads)
        k_gen = self._project_and_reshape(gen_tokens, self.add_k_proj, self.num_kv_heads)
        v_gen = self._project_and_reshape(gen_tokens, self.add_v_proj, self.num_kv_heads)

        # QK normalization
        q_gen = self.norm_added_q(q_gen)
        k_gen = self.norm_added_k(k_gen)

        # Apply RoPE to generation tokens
        gen_position_ids = position_ids[:, :, und_len : und_len + gen_len]
        cos, sin = self.rope(gen_position_ids, seq_len=gen_len)
        q_gen, k_gen = apply_rotary_pos_emb(q_gen, k_gen, cos, sin)

        # Concatenate [und + gen] keys/values for full attention
        k_full = mx.concatenate([und_keys, k_gen], axis=1)
        v_full = mx.concatenate([und_values, v_gen], axis=1)

        # GQA expansion
        if self.num_kv_heads != self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k_full = mx.repeat(k_full, repeat_factor, axis=2)
            v_full = mx.repeat(v_full, repeat_factor, axis=2)

        # Full (non-causal) attention
        q_t = mx.transpose(q_gen, (0, 2, 1, 3))
        k_t = mx.transpose(k_full, (0, 2, 1, 3))
        v_t = mx.transpose(v_full, (0, 2, 1, 3))

        attn_out = mx.fast.scaled_dot_product_attention(
            q_t, k_t, v_t, scale=self.scale
        )

        attn_out = mx.transpose(attn_out, (0, 2, 1, 3))
        attn_out = attn_out.reshape(batch, gen_len, self.num_heads * self.head_dim)
        return self.to_add_out(attn_out)
