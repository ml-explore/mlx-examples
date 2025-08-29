# MLX implementation of model.py
import math
from typing import List, Tuple, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.astype(mx.float32)

    # calculation
    arange_vals = mx.arange(half).astype(mx.float32)
    div_term = mx.power(10000, -arange_vals / half)
    sinusoid = position[:, None] @ div_term[None, :]
    x = mx.concatenate([mx.cos(sinusoid), mx.sin(sinusoid)], axis=1)
    return x


def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    positions = mx.arange(max_seq_len).astype(mx.float32)
    freqs = mx.arange(0, dim, 2).astype(mx.float32) / dim
    freqs = 1.0 / mx.power(theta, freqs)
    angles = positions[:, None] @ freqs[None, :]
    # Store as [max_seq_len, dim//2, 2] where last dimension is [real, imag]
    freqs_complex = mx.stack([mx.cos(angles), mx.sin(angles)], axis=-1)
    return freqs_complex


def rope_apply(x, grid_sizes, freqs):
    n, c = x.shape[2], x.shape[3] // 2

    # split freqs based on dimension allocation
    split_sizes = [c - 2 * (c // 3), c // 3, c // 3]
    freqs_splits = []
    start = 0
    for size in split_sizes:
        freqs_splits.append(freqs[:, start:start+size, :])
        start += size

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # reshape x_i to complex representation
        x_i = x[i, :seq_len].reshape(seq_len, n, c, 2)
        
        # precompute frequency multipliers for each dimension
        freqs_f = freqs_splits[0][:f].reshape(f, 1, 1, -1, 2)
        freqs_f = mx.tile(freqs_f, (1, h, w, 1, 1)).reshape(f * h * w, -1, 2)
        
        freqs_h = freqs_splits[1][:h].reshape(1, h, 1, -1, 2) 
        freqs_h = mx.tile(freqs_h, (f, 1, w, 1, 1)).reshape(f * h * w, -1, 2)
        
        freqs_w = freqs_splits[2][:w].reshape(1, 1, w, -1, 2)
        freqs_w = mx.tile(freqs_w, (f, h, 1, 1, 1)).reshape(f * h * w, -1, 2)
        
        # Concatenate frequency components
        freqs_i = mx.concatenate([freqs_f, freqs_h, freqs_w], axis=1)
        freqs_i = freqs_i[:seq_len].reshape(seq_len, 1, c, 2)

        # apply rotary embedding (complex multiplication)
        x_real = x_i[..., 0]
        x_imag = x_i[..., 1]
        freqs_real = freqs_i[..., 0]
        freqs_imag = freqs_i[..., 1]
        
        out_real = x_real * freqs_real - x_imag * freqs_imag
        out_imag = x_real * freqs_imag + x_imag * freqs_real
        
        x_i = mx.stack([out_real, out_imag], axis=-1).reshape(seq_len, n, -1)
        
        # Handle remaining sequence
        if x.shape[1] > seq_len:
            x_i = mx.concatenate([x_i, x[i, seq_len:]], axis=0)

        output.append(x_i)
    
    return mx.stack(output)


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x):
        """
        Args:
            x(Array): Shape [B, L, C]
        """
        return self._norm(x) * self.weight

    def _norm(self, x):
        return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, affine=False):
        super().__init__(dims=dim, eps=eps, affine=affine)

    def __call__(self, x):
        """
        Args:
            x(Array): Shape [B, L, C]
        """
        return super().__call__(x)


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

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def __call__(self, x, seq_lens, grid_sizes, freqs):
        """
        Args:
            x(Array): Shape [B, L, C]
            seq_lens(Array): Shape [B]
            grid_sizes(Array): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Array): Rope freqs, shape [1024, C / num_heads / 2, 2]
        """
        b, s, n, d = x.shape[0], x.shape[1], self.num_heads, self.head_dim

        # query, key, value function
        q = self.norm_q(self.q(x)).reshape(b, s, n, d)
        k = self.norm_k(self.k(x)).reshape(b, s, n, d)
        v = self.v(x).reshape(b, s, n, d)

        x = mlx_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.reshape(b, s, -1)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):

    def __call__(self, x, context, context_lens):
        """
        Args:
            x(Array): Shape [B, L1, C]
            context(Array): Shape [B, L2, C]
            context_lens(Array): Shape [B]
        """
        b, n, d = x.shape[0], self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).reshape(b, -1, n, d)
        k = self.norm_k(self.k(context)).reshape(b, -1, n, d)
        v = self.v(context).reshape(b, -1, n, d)

        # compute attention
        x = mlx_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.reshape(b, -1, self.dim)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm,
                                            eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), 
            nn.GELU(),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = mx.random.normal((1, 6, dim)) / dim**0.5

    def __call__(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        """
        Args:
            x(Array): Shape [B, L, C]
            e(Array): Shape [B, L1, 6, C]
            seq_lens(Array): Shape [B], length of each sequence in batch
            grid_sizes(Array): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Array): Rope freqs, shape [1024, C / num_heads / 2, 2]
        """
        e = mx.split(self.modulation + e, 6, axis=2)

        # self-attention
        y = self.self_attn(
            self.norm1(x) * (1 + mx.squeeze(e[1], axis=2)) + mx.squeeze(e[0], axis=2),
            seq_lens, grid_sizes, freqs)
        x = x + y * mx.squeeze(e[2], axis=2)

        # cross-attention & ffn function
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn(
            self.norm2(x) * (1 + mx.squeeze(e[4], axis=2)) + mx.squeeze(e[3], axis=2))
        x = x + y * mx.squeeze(e[5], axis=2)
        
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = mx.random.normal((1, 2, dim)) / dim**0.5

    def __call__(self, x, e):
        """
        Args:
            x(Array): Shape [B, L1, C]
            e(Array): Shape [B, L1, C]
        """
        e = mx.split(self.modulation + mx.expand_dims(e, axis=2), 2, axis=2)
        x = self.head(
            self.norm(x) * (1 + mx.squeeze(e[1], axis=2)) + mx.squeeze(e[0], axis=2))
        return x


class WanModel(nn.Module):
    """
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        """
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v']
        self.model_type = model_type

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

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), 
            nn.GELU(),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), 
            nn.SiLU(), 
            nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = [
            WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps) for _ in range(num_layers)
        ]

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = mx.concatenate([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], axis=1)

        # initialize weights
        self.init_weights()

    def __call__(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
    ):
        """
        Forward pass through the diffusion model

        Args:
            x (List[Array]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Array):
                Diffusion timesteps tensor of shape [B]
            context (List[Array]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Array], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Array]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert y is not None

        if y is not None:
            x = [mx.concatenate([u, v], axis=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(mx.expand_dims(mx.transpose(u, (1, 2, 3, 0)), axis=0)) for u in x]

        grid_sizes = mx.stack(
            [mx.array(u.shape[1:4], dtype=mx.int32) for u in x])
        

        x = [u.reshape(u.shape[0], -1, u.shape[-1]) for u in x]

        seq_lens = mx.array([u.shape[1] for u in x], dtype=mx.int32)
        assert seq_lens.max() <= seq_len
        
        # Pad sequences
        x_padded = []
        for u in x:
            pad_len = seq_len - u.shape[1]
            if pad_len > 0:
                padding = mx.zeros((u.shape[0], pad_len, u.shape[2]))
                u = mx.concatenate([u, padding], axis=1)
            x_padded.append(u)
        x = mx.concatenate(x_padded, axis=0)

        # time embeddings
        if t.ndim == 1:
            t = mx.broadcast_to(t[:, None], (t.shape[0], seq_len))
        bt = t.shape[0]
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).reshape(bt, seq_len, -1))
        e0 = self.time_projection(e).reshape(bt, seq_len, 6, self.dim)

        # context
        context_lens = None
        context_padded = []
        for u in context:
            pad_len = self.text_len - u.shape[0]
            if pad_len > 0:
                padding = mx.zeros((pad_len, u.shape[1]))
                u = mx.concatenate([u, padding], axis=0)
            context_padded.append(u)
        context = self.text_embedding(mx.stack(context_padded))

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)

        for block in self.blocks:
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x

    def unpatchify(self, x, grid_sizes):
        """
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Array]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Array):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Array]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for i, v in enumerate(grid_sizes):
            v = v.tolist()
            seq_len = math.prod(v)
            u = x[i, :seq_len].reshape(*v, *self.patch_size, c)
            # Rearrange dimensions: (f, h, w, p, q, r, c) -> (c, f*p, h*q, w*r)
            u = mx.transpose(u, (6, 0, 3, 1, 4, 2, 5))
            u = u.reshape(c, v[0] * self.patch_size[0], 
                          v[1] * self.patch_size[1], 
                          v[2] * self.patch_size[2])
            out.append(u)
        return out

    def init_weights(self):
        """
        Initialize model parameters using Xavier initialization.
        """
        # Initialize patch embedding
        fan_in = self.in_dim * math.prod(self.patch_size)
        fan_out = self.dim
        std = math.sqrt(2.0 / (fan_in + fan_out))
        self.patch_embedding.weight = mx.random.uniform(
            low=-std, high=std, shape=self.patch_embedding.weight.shape)
        
        # Initialize text embedding layers with normal distribution
        text_layers = list(self.text_embedding.layers)
        for i in [0, 2]:  # First and third layers
            layer = text_layers[i]
            layer.weight = mx.random.normal(shape=layer.weight.shape) * 0.02
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias = mx.zeros(layer.bias.shape)
        
        # Initialize time embedding layers
        time_layers = list(self.time_embedding.layers)
        for i in [0, 2]:  # First and third layers
            layer = time_layers[i]
            layer.weight = mx.random.normal(shape=layer.weight.shape) * 0.02
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias = mx.zeros(layer.bias.shape)
        
        # Initialize output head to zeros
        self.head.head.weight = mx.zeros(self.head.head.weight.shape)
        if hasattr(self.head.head, 'bias') and self.head.head.bias is not None:
            self.head.head.bias = mx.zeros(self.head.head.bias.shape)