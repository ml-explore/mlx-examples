from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .attention import MultiheadAttention


def symmetrize(x: mx.array) -> mx.array:
    """
    Make a tensor symmetric over its last two dimensions.

    Args:
        x: Tensor with shape (..., L, L).

    Returns:
        mx.array: Symmetrized tensor of shape (..., L, L).
    """
    # Add tensor to its transpose over the last two dims
    return x + mx.swapaxes(x, -1, -2)


def apc(x: mx.array) -> mx.array:
    """
    Apply Average Product Correction (APC) to remove background co-variation.

    Args:
        x: Tensor with shape (..., L, L).

    Returns:
        mx.array: APC-corrected tensor of shape (..., L, L).
    """
    # Compute row, column, and total sums
    a1 = mx.sum(x, axis=-1, keepdims=True)
    a2 = mx.sum(x, axis=-2, keepdims=True)
    a12 = mx.sum(x, axis=(-1, -2), keepdims=True)

    # Expected co-variation under independence
    expected = (a1 * a2) / a12
    return x - expected


class TransformerLayer(nn.Module):
    """
    Transformer layer used in ESM-2.

    Args:
        embed_dim (int): Model embedding dimension.
        ffn_embed_dim (int): Hidden dimension of the feed-forward network.
        attention_heads (int): Number of attention heads.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        attention_heads: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self._init_submodules()

    def _init_submodules(self) -> None:
        """Initialize attention, norms, and feed-forward submodules."""
        self.self_attn = MultiheadAttention(self.embed_dim, self.attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def __call__(
        self,
        x: mx.array,
        self_attn_mask: Optional[mx.array] = None,
        self_attn_padding_mask: Optional[mx.array] = None,
        need_head_weights: bool = False,
    ):
        """
        Forward pass for the Transformer layer.

        Args:
            x: Tensor of shape (seq_len, batch, embed_dim).
            self_attn_mask: Optional attention mask.
            self_attn_padding_mask: Optional padding mask of shape (batch, seq_len).
            need_head_weights: If True, return per-head attention weights.

        Returns:
            x: Tensor of shape (seq_len, batch, embed_dim).
            attn: Attention weights of shape
                (num_heads, batch, tgt_len, src_len) if per-head,
                or (batch, tgt_len, src_len) if averaged.
        """
        # Self-attention block
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key_padding_mask=self_attn_padding_mask,
            attn_mask=self_attn_mask,
            need_head_weights=need_head_weights,
        )
        x = residual + x

        # Feed-forward block
        residual = x
        x = self.final_layer_norm(x)
        x = nn.gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x, attn


class RobertaLMHead(nn.Module):
    """
    Masked Language Modeling (MLM) head with tied weights.

    Args:
        embed_dim (int): Embedding dimension of the backbone.
        output_dim (int): Vocabulary size.
        weight (mx.array): Embedding weight matrix for tied projection.
    """

    def __init__(self, embed_dim: int, output_dim: int, weight: mx.array):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.weight = weight
        self.bias = mx.zeros(output_dim)

    def __call__(self, features: mx.array) -> mx.array:
        """
        Forward pass for the MLM head.

        Args:
            features: Tensor of shape (seq_len, batch, embed_dim).

        Returns:
            mx.array: Logits of shape (seq_len, batch, output_dim).
        """
        # Transform features before projection to vocab
        x = self.dense(features)
        x = nn.gelu(x)
        x = self.layer_norm(x)
        return mx.matmul(x, self.weight.T) + self.bias


class ContactPredictionHead(nn.Module):
    """
    Predict residue-residue contact probabilities from attention maps.

    Args:
        in_features (int): Number of attention channels (layers × heads).
        prepend_bos (bool): If True, drop BOS/CLS token attentions.
        append_eos (bool): If True, drop EOS token attentions.
        bias (bool): Whether the regression layer uses a bias term.
        eos_idx (Optional[int]): Token ID for EOS; required if append_eos=True.
    """

    def __init__(
        self,
        in_features: int,
        prepend_bos: bool,
        append_eos: bool,
        bias: bool = True,
        eos_idx: Optional[int] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        if append_eos and eos_idx is None:
            raise ValueError("append_eos=True but eos_idx was not provided.")
        self.eos_idx = eos_idx
        self.regression = nn.Linear(in_features, 1, bias=bias)

    def __call__(self, tokens: mx.array, attentions: mx.array) -> mx.array:
        """
        Forward pass for contact prediction.

        Args:
            tokens: Tensor of shape (B, T).
            attentions: Tensor of shape (B, L, H, T, T).

        Returns:
            mx.array: Contact probabilities of shape (B, T', T'),
                where T' = T - [prepend_bos] - [append_eos].
        """
        # Remove EOS attentions if requested
        if self.append_eos:
            eos_mask = mx.not_equal(tokens, self.eos_idx).astype(attentions.dtype)
            eos_mask = eos_mask[:, None, :] * eos_mask[:, :, None]
            attentions = attentions * eos_mask[:, None, None, :, :]
            attentions = attentions[..., :-1, :-1]

        # Remove BOS attentions if requested
        if self.prepend_bos:
            attentions = attentions[..., 1:, 1:]

        # Merge (layers × heads) into channel dimension
        batch_size, layers, heads, seqlen, _ = attentions.shape
        attentions = attentions.reshape(batch_size, layers * heads, seqlen, seqlen)

        # Symmetrize and apply APC to enhance contact signal
        attentions = apc(symmetrize(attentions))

        # Apply logistic regression over channels
        attentions = mx.transpose(attentions, axes=[0, 2, 3, 1])
        logits = self.regression(attentions)
        return nn.sigmoid(mx.squeeze(logits, axis=3))
