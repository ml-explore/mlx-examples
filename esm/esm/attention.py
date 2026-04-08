from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .rotary_embedding import RotaryEmbedding


class MultiheadAttention(nn.Module):
    """
    Multi-head attention layer with rotary position embeddings, as used in ESM-2.

    This module implements both self-attention (when `key` and `value` are not
    provided) and cross-attention. It projects input sequences into queries,
    keys, and values, applies rotary position embeddings to encode relative
    position information, computes scaled dot-product attention over multiple
    heads in parallel, and returns a combined output projection.

    Args:
        embed_dim (int): Total embedding dimension of the model input and output.
        num_heads (int): Number of parallel attention heads. Must divide `embed_dim`.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        # Linear projections for queries, keys, and values (with bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # Linear projection for output (with bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # ESM-2 uses rotary embeddings
        self.rot_emb = RotaryEmbedding(dim=self.head_dim)

    def __call__(
        self,
        query,
        key: Optional[mx.array] = None,
        value: Optional[mx.array] = None,
        key_padding_mask: Optional[mx.array] = None,
        attn_mask: Optional[mx.array] = None,
        need_head_weights: bool = False,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """
        Multi-head attention forward pass.

        Args:
            query: Tensor of shape (tgt_len, batch, embed_dim).
            key: Optional tensor of shape (src_len, batch, embed_dim). Defaults to `query`.
            value: Optional tensor of shape (src_len, batch, embed_dim). Defaults to `query`.
            key_padding_mask: Optional mask of shape (batch, src_len) to ignore padded positions.
            attn_mask: Optional mask for attention (e.g., causal mask).
            need_head_weights: If True, return attention weights for each head separately.

        Returns:
            attn_output: Tensor of shape (tgt_len, batch, embed_dim).
            attn_weights_out: Attention weights of shape
                (num_heads, batch, tgt_len, src_len) if per-head,
                or (batch, tgt_len, src_len) if averaged.
        """

        tgt_len, bsz, embed_dim = query.shape
        assert embed_dim == self.embed_dim

        # For self-attention, use query as key and value if not provided
        if key is None:
            key = query
        if value is None:
            value = query

        # Project queries, keys, values
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q * self.scaling

        # Reshape for multi-head attention
        q = q.reshape(tgt_len, bsz * self.num_heads, self.head_dim).swapaxes(0, 1)
        k = k.reshape(-1, bsz * self.num_heads, self.head_dim).swapaxes(0, 1)
        v = v.reshape(-1, bsz * self.num_heads, self.head_dim).swapaxes(0, 1)

        src_len = k.shape[1]

        # Apply rotary embeddings if present
        if self.rot_emb:
            q, k = self.rot_emb(q, k)

        # Compute attention weights
        attn_weights = q @ k.swapaxes(-2, -1)

        assert list(attn_weights.shape) == [bsz * self.num_heads, tgt_len, src_len]

        # Apply attention mask
        if attn_mask is not None:
            attn_mask = mx.expand_dims(attn_mask, 0)
            attn_weights = attn_weights + attn_mask

        # Apply key padding mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.reshape(bsz, self.num_heads, tgt_len, src_len)
            # Convert key_padding_mask to boolean and expand dimensions
            # key_padding_mask: [bsz, src_len] -> [bsz, 1, 1, src_len]
            mask = mx.expand_dims(
                mx.expand_dims(key_padding_mask.astype(mx.bool_), 1), 2
            )
            # Apply mask: set attention to -inf where mask is True (padded positions)
            attn_weights = mx.where(mask, -mx.inf, attn_weights)
            attn_weights = attn_weights.reshape(bsz * self.num_heads, tgt_len, src_len)

        # Apply softmax
        attn_weights_float = mx.softmax(attn_weights.astype(mx.float32), axis=-1)
        attn_probs = attn_weights_float

        # Compute attention output
        attn = attn_probs @ v
        assert list(attn.shape) == [bsz * self.num_heads, tgt_len, self.head_dim]

        # Reshape output
        attn = attn.swapaxes(0, 1).reshape(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # Return attention weights if requested
        attn_weights_out: Optional[mx.array] = None
        if need_head_weights:
            # Return attention weights for each head separately
            attn_weights_out = (
                attn_weights_float.reshape(bsz, self.num_heads, tgt_len, src_len)
                .astype(attn.dtype)
                .swapaxes(0, 1)
            )
        else:
            # Return averaged attention weights
            attn_weights_out = mx.mean(
                attn_weights_float.reshape(bsz, self.num_heads, tgt_len, src_len),
                axis=1,
            ).astype(attn.dtype)

        return attn, attn_weights_out
