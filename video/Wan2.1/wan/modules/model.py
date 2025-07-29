# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# MLX Implementation of WAN Model - True 1:1 Port from PyTorch

import math
from typing import List, Tuple, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim: int, position: mx.array) -> mx.array:
    """Generate sinusoidal position embeddings."""
    assert dim % 2 == 0
    half = dim // 2
    position = position.astype(mx.float32)
    
    # Calculate sinusoidal embeddings
    div_term = mx.power(10000, mx.arange(half).astype(mx.float32) / half)
    sinusoid = mx.expand_dims(position, 1) / mx.expand_dims(div_term, 0)
    
    return mx.concatenate([mx.cos(sinusoid), mx.sin(sinusoid)], axis=1)


def rope_params(max_seq_len: int, dim: int, theta: float = 10000) -> mx.array:
    """Generate RoPE (Rotary Position Embedding) parameters."""
    assert dim % 2 == 0
    positions = mx.arange(max_seq_len)
    freqs = mx.arange(0, dim, 2).astype(mx.float32) / dim
    freqs = 1.0 / mx.power(theta, freqs)
    
    # Outer product
    freqs = mx.expand_dims(positions, 1) * mx.expand_dims(freqs, 0)
    
    # Convert to complex representation
    return mx.stack([mx.cos(freqs), mx.sin(freqs)], axis=-1)


def rope_apply(x: mx.array, grid_sizes: mx.array, freqs: mx.array) -> mx.array:
    """Apply rotary position embeddings to input tensor."""
    n, c_half = x.shape[2], x.shape[3] // 2
    
    # Split frequencies for different dimensions
    c_split = c_half - 2 * (c_half // 3)
    freqs_splits = [
        freqs[:, :c_split],
        freqs[:, c_split:c_split + c_half // 3],
        freqs[:, c_split + c_half // 3:]
    ]
    
    output = []
    for i in range(grid_sizes.shape[0]):
        f, h, w = int(grid_sizes[i, 0]), int(grid_sizes[i, 1]), int(grid_sizes[i, 2])
        seq_len = f * h * w
        
        # Extract sequence for current sample
        x_i = x[i, :seq_len].astype(mx.float32)
        x_i = x_i.reshape(seq_len, n, -1, 2)
        
        # Prepare frequency tensors
        freqs_f = freqs_splits[0][:f].reshape(f, 1, 1, -1, 2)
        freqs_f = mx.broadcast_to(freqs_f, (f, h, w, freqs_f.shape[-2], 2))
        
        freqs_h = freqs_splits[1][:h].reshape(1, h, 1, -1, 2)
        freqs_h = mx.broadcast_to(freqs_h, (f, h, w, freqs_h.shape[-2], 2))
        
        freqs_w = freqs_splits[2][:w].reshape(1, 1, w, -1, 2)
        freqs_w = mx.broadcast_to(freqs_w, (f, h, w, freqs_w.shape[-2], 2))
        
        # Concatenate and reshape frequencies
        freqs_i = mx.concatenate([freqs_f, freqs_h, freqs_w], axis=-2)
        freqs_i = freqs_i.reshape(seq_len, 1, -1, 2)
        
        # Apply rotary embedding
        x_real = x_i[..., 0]
        x_imag = x_i[..., 1]
        freqs_cos = freqs_i[..., 0]
        freqs_sin = freqs_i[..., 1]
        
        x_rotated_real = x_real * freqs_cos - x_imag * freqs_sin
        x_rotated_imag = x_real * freqs_sin + x_imag * freqs_cos
        
        x_i = mx.stack([x_rotated_real, x_rotated_imag], axis=-1).reshape(seq_len, n, -1)
        
        # Concatenate with remaining sequence if any
        if seq_len < x.shape[1]:
            x_i = mx.concatenate([x_i, x[i, seq_len:]], axis=0)
        
        output.append(x_i)
    
    return mx.stack(output).astype(x.dtype)


class WanRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = mx.ones((dim,))
    
    def __call__(self, x: mx.array) -> mx.array:
        # RMS normalization
        variance = mx.mean(mx.square(x.astype(mx.float32)), axis=-1, keepdims=True)
        x_normed = x * mx.rsqrt(variance + self.eps)
        return (x_normed * self.weight).astype(x.dtype)


class WanLayerNorm(nn.Module):
    """Layer normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = mx.ones((dim,))
            self.bias = mx.zeros((dim,))
    
    def __call__(self, x: mx.array) -> mx.array:
        # Standard layer normalization
        x_float = x.astype(mx.float32)
        mean = mx.mean(x_float, axis=-1, keepdims=True)
        variance = mx.var(x_float, axis=-1, keepdims=True)
        x_normed = (x_float - mean) * mx.rsqrt(variance + self.eps)
        
        if self.elementwise_affine:
            x_normed = x_normed * self.weight + self.bias
        
        return x_normed.astype(x.dtype)


def mlx_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    q_lens: Optional[mx.array] = None,
    k_lens: Optional[mx.array] = None,
    dropout_p: float = 0.,
    softmax_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    dtype: Optional[type] = None,
) -> mx.array:
    """
    MLX implementation of scaled dot-product attention.
    """
    # Get shapes
    b, lq, n, d = q.shape
    _, lk, _, _ = k.shape
    
    # Scale queries if needed
    if q_scale is not None:
        q = q * q_scale
    
    # Compute attention scores
    q = q.transpose(0, 2, 1, 3)  # [b, n, lq, d]
    k = k.transpose(0, 2, 1, 3)  # [b, n, lk, d]
    v = v.transpose(0, 2, 1, 3)  # [b, n, lk, d]
    
    # Compute attention scores
    scores = mx.matmul(q, k.transpose(0, 1, 3, 2))  # [b, n, lq, lk]
    
    # Apply softmax scale if provided
    if softmax_scale is not None:
        scores = scores * softmax_scale
    else:
        # Default scaling by sqrt(d)
        scores = scores / mx.sqrt(mx.array(d, dtype=scores.dtype))
    
    # Create attention mask
    attn_mask = None
    
    # Apply window size masking if specified
    if window_size != (-1, -1):
        left_window, right_window = window_size
        window_mask = mx.zeros((lq, lk))
        for i in range(lq):
            start = max(0, i - left_window)
            end = min(lk, i + right_window + 1)
            window_mask[i, start:end] = 1
        attn_mask = window_mask
    
    # Apply causal masking if needed
    if causal:
        causal_mask = mx.tril(mx.ones((lq, lk)), k=0)
        if attn_mask is None:
            attn_mask = causal_mask
        else:
            attn_mask = mx.logical_and(attn_mask, causal_mask)
    
    # Apply attention mask if present
    if attn_mask is not None:
        attn_mask = attn_mask.astype(scores.dtype)
        scores = scores * attn_mask + (1 - attn_mask) * -1e4
    
    # Apply attention mask if lengths are provided
    if q_lens is not None or k_lens is not None:
        if q_lens is not None:
            mask = mx.arange(lq)[None, :] < q_lens[:, None]
            mask = mask.astype(scores.dtype)
            scores = scores * mask[:, None, :, None] + (1 - mask[:, None, :, None]) * -1e4
        if k_lens is not None:
            mask = mx.arange(lk)[None, :] < k_lens[:, None]
            mask = mask.astype(scores.dtype)
            scores = scores * mask[:, None, None, :] + (1 - mask[:, None, None, :]) * -1e4
    
    # Apply softmax
    max_scores = mx.max(scores, axis=-1, keepdims=True)
    scores = scores - max_scores
    exp_scores = mx.exp(scores)
    sum_exp = mx.sum(exp_scores, axis=-1, keepdims=True)
    attn = exp_scores / (sum_exp + 1e-6)
    
    # Apply dropout if needed
    if dropout_p > 0 and not deterministic:
        raise NotImplementedError("Dropout not implemented in MLX version")
    
    # Compute output
    out = mx.matmul(attn, v)  # [b, n, lq, d]
    out = out.transpose(0, 2, 1, 3)  # [b, lq, n, d]
    
    return out


class WanSelfAttention(nn.Module):
    """Self-attention module with RoPE and optional QK normalization."""
    
    def __init__(self, dim: int, num_heads: int, window_size: Tuple[int, int] = (-1, -1),
                 qk_norm: bool = True, eps: float = 1e-6):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        
        # Linear projections
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        
        # Normalization layers
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
    
    def __call__(self, x: mx.array, seq_lens: mx.array, grid_sizes: mx.array, 
                 freqs: mx.array) -> mx.array:
        b, s = x.shape[0], x.shape[1]
        
        # Compute Q, K, V
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        
        if self.qk_norm:
            q = self.norm_q(q)
            k = self.norm_k(k)
        
        # Reshape for multi-head attention
        q = q.reshape(b, s, self.num_heads, self.head_dim)
        k = k.reshape(b, s, self.num_heads, self.head_dim)
        v = v.reshape(b, s, self.num_heads, self.head_dim)
        
        # Apply RoPE
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
        
        # Apply attention
        x = mlx_attention(q, k, v, k_lens=seq_lens, window_size=self.window_size)
        
        # Reshape and project output
        x = x.reshape(b, s, self.dim)
        x = self.o(x)
        
        return x


class WanT2VCrossAttention(WanSelfAttention):
    """Text-to-video cross attention."""
    
    def __call__(self, x: mx.array, context: mx.array, context_lens: mx.array) -> mx.array:
        b = x.shape[0]
        
        # Compute queries from x
        q = self.q(x)
        if self.qk_norm:
            q = self.norm_q(q)
        q = q.reshape(b, -1, self.num_heads, self.head_dim)
        
        # Compute keys and values from context
        k = self.k(context)
        v = self.v(context)
        if self.qk_norm:
            k = self.norm_k(k)
        k = k.reshape(b, -1, self.num_heads, self.head_dim)
        v = v.reshape(b, -1, self.num_heads, self.head_dim)
        
        # Apply attention
        x = mlx_attention(q, k, v, k_lens=context_lens)
        
        # Reshape and project output
        x = x.reshape(b, -1, self.dim)
        x = self.o(x)
        
        return x


class WanI2VCrossAttention(WanSelfAttention):
    """Image-to-video cross attention."""
    
    def __init__(self, dim: int, num_heads: int, window_size: Tuple[int, int] = (-1, -1),
                 qk_norm: bool = True, eps: float = 1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)
        
        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
    
    def __call__(self, x: mx.array, context: mx.array, context_lens: mx.array) -> mx.array:
        # Split context into image and text parts
        context_img = context[:, :257]
        context = context[:, 257:]
        
        b = x.shape[0]
        
        # Compute queries
        q = self.q(x)
        if self.qk_norm:
            q = self.norm_q(q)
        q = q.reshape(b, -1, self.num_heads, self.head_dim)
        
        # Compute keys and values for text
        k = self.k(context)
        v = self.v(context)
        if self.qk_norm:
            k = self.norm_k(k)
        k = k.reshape(b, -1, self.num_heads, self.head_dim)
        v = v.reshape(b, -1, self.num_heads, self.head_dim)
        
        # Compute keys and values for image
        k_img = self.k_img(context_img)
        v_img = self.v_img(context_img)
        if self.qk_norm:
            k_img = self.norm_k_img(k_img)
        k_img = k_img.reshape(b, -1, self.num_heads, self.head_dim)
        v_img = v_img.reshape(b, -1, self.num_heads, self.head_dim)
        
        # Apply attention
        img_x = mlx_attention(q, k_img, v_img, k_lens=None)
        x = mlx_attention(q, k, v, k_lens=context_lens)
        
        # Combine and project
        img_x = img_x.reshape(b, -1, self.dim)
        x = x.reshape(b, -1, self.dim)
        x = x + img_x
        x = self.o(x)
        
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):
    """Transformer block with self-attention, cross-attention, and FFN."""
    
    def __init__(self, cross_attn_type: str, dim: int, ffn_dim: int, num_heads: int,
                 window_size: Tuple[int, int] = (-1, -1), qk_norm: bool = True,
                 cross_attn_norm: bool = False, eps: float = 1e-6):
        super().__init__()
        
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        
        # Layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps)
        
        self.norm2 = WanLayerNorm(dim, eps)
        
        # FFN - use a list instead of Sequential to match PyTorch exactly!
        self.ffn = [
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim)
        ]
        
        # Modulation parameters
        self.modulation = mx.random.normal((1, 6, dim)) / math.sqrt(dim)
    
    def __call__(self, x: mx.array, e: mx.array, seq_lens: mx.array, 
                 grid_sizes: mx.array, freqs: mx.array, context: mx.array,
                 context_lens: Optional[mx.array]) -> mx.array:
        # Apply modulation
        e = (self.modulation + e).astype(mx.float32)
        e_chunks = [mx.squeeze(chunk, axis=1) for chunk in mx.split(e, 6, axis=1)]
        
        # Self-attention with modulation
        y = self.norm1(x).astype(mx.float32)
        y = y * (1 + e_chunks[1]) + e_chunks[0]
        y = self.self_attn(y, seq_lens, grid_sizes, freqs)
        x = x + y * e_chunks[2]
        
        # Cross-attention
        if self.cross_attn_norm and isinstance(self.norm3, WanLayerNorm):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
        else:
            x = x + self.cross_attn(x, context, context_lens)
        
        # FFN with modulation
        y = self.norm2(x).astype(mx.float32)
        y = y * (1 + e_chunks[4]) + e_chunks[3]
        
        # Apply FFN layers manually
        y = self.ffn[0](y)  # Linear
        y = self.ffn[1](y)  # GELU
        y = self.ffn[2](y)  # Linear
        
        x = x + y * e_chunks[5]
        
        return x


class Head(nn.Module):
    """Output head for final projection."""
    
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], 
                 eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        
        # Output projection
        out_features = int(np.prod(patch_size)) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_features)
        
        # Modulation
        self.modulation = mx.random.normal((1, 2, dim)) / math.sqrt(dim)
    
    def __call__(self, x: mx.array, e: mx.array) -> mx.array:
        # Apply modulation
        e = (self.modulation + mx.expand_dims(e, 1)).astype(mx.float32)
        e_chunks = mx.split(e, 2, axis=1)
        
        # Apply normalization and projection with modulation
        x = self.norm(x) * (1 + e_chunks[1]) + e_chunks[0]
        x = self.head(x)
        
        return x


class MLPProj(nn.Module):
    """MLP projection for image embeddings."""
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        
        # Use a list to match PyTorch Sequential indexing
        self.proj = [
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        ]
    
    def __call__(self, image_embeds: mx.array) -> mx.array:
        x = image_embeds
        for layer in self.proj:
            x = layer(x)
        return x


class WanModel(nn.Module):
    """
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    MLX implementation - True 1:1 port from PyTorch.
    """
    
    def __init__(
        self,
        model_type: str = 't2v',
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 16,
        dim: int = 2048,
        ffn_dim: int = 8192,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 16,
        num_layers: int = 32,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6
    ):
        super().__init__()
        
        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type
        
        # Store configuration
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        
        # Embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Use lists instead of Sequential to match PyTorch!
        self.text_embedding = [
            nn.Linear(text_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        ]
        
        self.time_embedding = [
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        ]
        
        self.time_projection = [
            nn.SiLU(),
            nn.Linear(dim, dim * 6)
        ]
        
        # Transformer blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = [
            WanAttentionBlock(
                cross_attn_type, dim, ffn_dim, num_heads,
                window_size, qk_norm, cross_attn_norm, eps
            )
            for _ in range(num_layers)
        ]
        
        # Output head
        self.head = Head(dim, out_dim, patch_size, eps)
        
        # Precompute RoPE frequencies
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = mx.concatenate([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], axis=1)
        
        # Image embedding for i2v
        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)
        
        # Initialize weights
        self.init_weights()
    
    def __call__(
        self,
        x: List[mx.array],
        t: mx.array,
        context: List[mx.array],
        seq_len: int,
        clip_fea: Optional[mx.array] = None,
        y: Optional[List[mx.array]] = None
    ) -> List[mx.array]:
        """
        Forward pass through the diffusion model.
        
        Args:
            x: List of input video tensors [C_in, F, H, W]
            t: Diffusion timesteps [B]
            context: List of text embeddings [L, C]
            seq_len: Maximum sequence length
            clip_fea: CLIP image features for i2v mode
            y: Conditional video inputs for i2v mode
        
        Returns:
            List of denoised video tensors [C_out, F, H/8, W/8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        
        # Concatenate conditional inputs if provided
        if y is not None:
            x = [mx.concatenate([u, v], axis=0) for u, v in zip(x, y)]
        
        # Patch embedding
        x = [mx.transpose(mx.expand_dims(u, 0), (0, 2, 3, 4, 1)) for u in x]
        x = [self.patch_embedding(u) for u in x]
        # Transpose back from MLX format (N, D, H, W, C) to (N, C, D, H, W) for the rest of the model
        x = [mx.transpose(u, (0, 4, 1, 2, 3)) for u in x]
        grid_sizes = mx.array([[u.shape[2], u.shape[3], u.shape[4]] for u in x])
        
        # Flatten spatial dimensions
        x = [mx.transpose(u.reshape(u.shape[0], u.shape[1], -1), (0, 2, 1)) for u in x]
        seq_lens = mx.array([u.shape[1] for u in x])
        
        # Pad sequences to max length
        x_padded = []
        for u in x:
            if u.shape[1] < seq_len:
                padding = mx.zeros((1, seq_len - u.shape[1], u.shape[2]))
                u = mx.concatenate([u, padding], axis=1)
            x_padded.append(u)
        x = mx.concatenate(x_padded, axis=0)
        
        # Time embeddings - apply layers manually
        e = sinusoidal_embedding_1d(self.freq_dim, t).astype(mx.float32)
        e = self.time_embedding[0](e)  # Linear
        e = self.time_embedding[1](e)  # SiLU
        e = self.time_embedding[2](e)  # Linear
        
        # Time projection
        e = self.time_projection[0](e)  # SiLU
        e0 = self.time_projection[1](e).reshape(-1, 6, self.dim)  # Linear
        
        # Process context
        context_lens = None
        context_padded = []
        for u in context:
            if u.shape[0] < self.text_len:
                padding = mx.zeros((self.text_len - u.shape[0], u.shape[1]))
                u = mx.concatenate([u, padding], axis=0)
            context_padded.append(u)
        context = mx.stack(context_padded)
        
        # Apply text embedding layers manually
        context = self.text_embedding[0](context)  # Linear
        context = self.text_embedding[1](context)  # GELU
        context = self.text_embedding[2](context)  # Linear
        
        # Add image embeddings for i2v
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = mx.concatenate([context_clip, context], axis=1)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(
                x, e0, seq_lens, grid_sizes, self.freqs,
                context, context_lens
            )
        
        # Apply output head
        x = self.head(x, e)
        
        # Unpatchify
        x = self.unpatchify(x, grid_sizes)
        
        return [u.astype(mx.float32) for u in x]
    
    def unpatchify(self, x: mx.array, grid_sizes: mx.array) -> List[mx.array]:
        """Reconstruct video tensors from patch embeddings."""
        c = self.out_dim
        out = []
        
        for i in range(grid_sizes.shape[0]):
            f, h, w = int(grid_sizes[i, 0]), int(grid_sizes[i, 1]), int(grid_sizes[i, 2])
            seq_len = f * h * w
            
            # Extract relevant sequence
            u = x[i, :seq_len]
            
            # Reshape to grid with patches
            pf, ph, pw = self.patch_size
            u = u.reshape(f, h, w, pf, ph, pw, c)
            
            # Rearrange dimensions
            u = mx.transpose(u, (6, 0, 3, 1, 4, 2, 5))
            
            # Combine patches
            u = u.reshape(c, f * pf, h * ph, w * pw)
            
            out.append(u)
        
        return out
    
    def init_weights(self):
        """Initialize model parameters using Xavier/He initialization."""
        # Note: MLX doesn't have nn.init like PyTorch, so we manually initialize
        
        # Helper function for Xavier uniform initialization
        def xavier_uniform(shape):
            bound = mx.sqrt(6.0 / (shape[0] + shape[1]))
            return mx.random.uniform(low=-bound, high=bound, shape=shape)
        
        # Initialize linear layers in blocks
        for block in self.blocks:
            # Self attention
            block.self_attn.q.weight = xavier_uniform(block.self_attn.q.weight.shape)
            block.self_attn.k.weight = xavier_uniform(block.self_attn.k.weight.shape)
            block.self_attn.v.weight = xavier_uniform(block.self_attn.v.weight.shape)
            block.self_attn.o.weight = xavier_uniform(block.self_attn.o.weight.shape)
            
            # Cross attention
            block.cross_attn.q.weight = xavier_uniform(block.cross_attn.q.weight.shape)
            block.cross_attn.k.weight = xavier_uniform(block.cross_attn.k.weight.shape)
            block.cross_attn.v.weight = xavier_uniform(block.cross_attn.v.weight.shape)
            block.cross_attn.o.weight = xavier_uniform(block.cross_attn.o.weight.shape)
            
            # FFN layers - now it's a list!
            block.ffn[0].weight = xavier_uniform(block.ffn[0].weight.shape)
            block.ffn[2].weight = xavier_uniform(block.ffn[2].weight.shape)
            
            # Modulation
            block.modulation = mx.random.normal(
                shape=(1, 6, self.dim),
                scale=1.0 / math.sqrt(self.dim)
            )
        
        # Special initialization for embeddings
        # Patch embedding - Xavier uniform
        weight_shape = self.patch_embedding.weight.shape
        fan_in = weight_shape[1] * np.prod(self.patch_size)
        fan_out = weight_shape[0]
        bound = mx.sqrt(6.0 / (fan_in + fan_out))
        self.patch_embedding.weight = mx.random.uniform(
            low=-bound,
            high=bound,
            shape=weight_shape
        )
        
        # Text embedding - normal distribution with std=0.02
        self.text_embedding[0].weight = mx.random.normal(shape=self.text_embedding[0].weight.shape, scale=0.02)
        self.text_embedding[2].weight = mx.random.normal(shape=self.text_embedding[2].weight.shape, scale=0.02)
        
        # Time embedding - normal distribution with std=0.02
        self.time_embedding[0].weight = mx.random.normal(shape=self.time_embedding[0].weight.shape, scale=0.02)
        self.time_embedding[2].weight = mx.random.normal(shape=self.time_embedding[2].weight.shape, scale=0.02)
        
        # Output head - initialize to zeros
        self.head.head.weight = mx.zeros(self.head.head.weight.shape)
        
        # Head modulation
        self.head.modulation = mx.random.normal(
            shape=(1, 2, self.dim),
            scale=1.0 / math.sqrt(self.dim)
        )
        
        # Initialize i2v specific layers if present
        if self.model_type == 'i2v':
            for i in [1, 3]:  # Linear layers in the proj list
                if isinstance(self.img_emb.proj[i], nn.Linear):
                    self.img_emb.proj[i].weight = xavier_uniform(self.img_emb.proj[i].weight.shape)