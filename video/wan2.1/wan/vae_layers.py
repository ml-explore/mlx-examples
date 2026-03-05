# Copyright © 2026 Apple Inc.

"""
Building blocks for Wan2.1 VAE.

All layers use channels-last format (NTHWC) as required by MLX.
"""

import mlx.core as mx
import mlx.nn as nn

CACHE_T = 2


def _normalize_tuple(value, n):
    if isinstance(value, int):
        return (value,) * n
    return tuple(value)


def create_cache_entry(x, existing_cache=None):
    t = x.shape[1]
    if t >= CACHE_T:
        return x[:, -CACHE_T:, :, :, :]
    else:
        cache_x = x[:, -t:, :, :, :]
        if existing_cache is not None:
            old_frames = existing_cache[:, -(CACHE_T - t) :, :, :, :]
            return mx.concatenate([old_frames, cache_x], axis=1)
        else:
            pad_t = CACHE_T - t
            zeros = mx.zeros((x.shape[0], pad_t, *x.shape[2:]), dtype=x.dtype)
            return mx.concatenate([zeros, cache_x], axis=1)


def write_cache(feat_cache, idx, x):
    t = x.shape[1]
    if t >= CACHE_T:
        feat_cache[idx] = x[:, -CACHE_T:, :, :, :]
    else:
        cache_x = x[:, -t:, :, :, :]
        if feat_cache[idx] is not None:
            old_frames = feat_cache[idx][:, -(CACHE_T - t) :, :, :, :]
            cache_x = mx.concatenate([old_frames, cache_x], axis=1)
        feat_cache[idx] = cache_x


class CausalConv3d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _normalize_tuple(kernel_size, 3)
        self.stride = _normalize_tuple(stride, 3)
        self.padding = _normalize_tuple(padding, 3)
        self._temporal_pad = self.padding[0] * 2
        self._spatial_pad_h = self.padding[1]
        self._spatial_pad_w = self.padding[2]

        scale = (
            1.0
            / (
                in_channels
                * self.kernel_size[0]
                * self.kernel_size[1]
                * self.kernel_size[2]
            )
            ** 0.5
        )
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, *self.kernel_size, in_channels),
        )
        if bias:
            self.bias = mx.zeros((out_channels,))

    def __call__(self, x, cache_x=None):
        temporal_pad = self._temporal_pad
        if cache_x is not None and self._temporal_pad > 0:
            x = mx.concatenate([cache_x, x], axis=1)
            temporal_pad = max(0, self._temporal_pad - cache_x.shape[1])

        if temporal_pad > 0:
            x = mx.pad(x, [(0, 0), (temporal_pad, 0), (0, 0), (0, 0), (0, 0)])

        if self._spatial_pad_h > 0 or self._spatial_pad_w > 0:
            x = mx.pad(
                x,
                [
                    (0, 0),
                    (0, 0),
                    (self._spatial_pad_h, self._spatial_pad_h),
                    (self._spatial_pad_w, self._spatial_pad_w),
                    (0, 0),
                ],
            )

        y = mx.conv3d(x, self.weight, stride=self.stride, padding=0)
        if "bias" in self:
            y = y + self.bias
        return y


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-12):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x):
        weight = self.weight
        if weight.ndim > 1:
            weight = weight.squeeze()
        return mx.fast.rms_norm(x, weight, self.eps)


class Upsample(nn.Module):
    def __init__(self, scale_factor=(2.0, 2.0)):
        super().__init__()
        self.scale_factor = scale_factor

    def __call__(self, x):
        return nn.Upsample(scale_factor=self.scale_factor, mode="nearest")(x)


class Resample(nn.Module):
    def __init__(self, dim, mode):
        assert mode in (
            "none",
            "upsample2d",
            "upsample3d",
            "downsample2d",
            "downsample3d",
        )
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == "upsample2d":
            self.upsample = Upsample()
            scale = 1.0 / (dim * 3 * 3) ** 0.5
            self.conv_weight = mx.random.uniform(
                low=-scale, high=scale, shape=(dim // 2, 3, 3, dim)
            )
            self.conv_bias = mx.zeros((dim // 2,))
        elif mode == "upsample3d":
            self.upsample = Upsample()
            scale = 1.0 / (dim * 3 * 3) ** 0.5
            self.conv_weight = mx.random.uniform(
                low=-scale, high=scale, shape=(dim // 2, 3, 3, dim)
            )
            self.conv_bias = mx.zeros((dim // 2,))
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode in ("downsample2d", "downsample3d"):
            scale = 1.0 / (dim * 3 * 3) ** 0.5
            self.conv_weight = mx.random.uniform(
                low=-scale, high=scale, shape=(dim, 3, 3, dim)
            )
            self.conv_bias = mx.zeros((dim,))
            if mode == "downsample3d":
                self.time_conv = CausalConv3d(
                    dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
                )

    def __call__(self, x, feat_cache, feat_idx):
        b, t, h, w, c = x.shape

        if self.mode == "upsample3d":
            if feat_cache[feat_idx] is None:
                feat_cache[feat_idx] = mx.zeros((b, CACHE_T, h, w, c), dtype=x.dtype)
                feat_idx += 1
            else:
                cache_input = x
                x = self.time_conv(x, feat_cache[feat_idx])
                write_cache(feat_cache, feat_idx, cache_input)
                feat_idx += 1
                x = x.reshape(b, t, h, w, 2, c)
                x = x.transpose(0, 1, 4, 2, 3, 5)
                x = x.reshape(b, t * 2, h, w, c)

        t_out = x.shape[1]
        c_out = x.shape[4]
        x = x.reshape(b * t_out, x.shape[2], x.shape[3], c_out)

        if self.mode in ("upsample2d", "upsample3d"):
            x = self.upsample(x)
            x = mx.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)])
            x = mx.conv2d(x, self.conv_weight, stride=1, padding=0)
            x = x + self.conv_bias
        elif self.mode in ("downsample2d", "downsample3d"):
            x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
            x = mx.conv2d(x, self.conv_weight, stride=2, padding=0)
            x = x + self.conv_bias

        x = x.reshape(b, t_out, x.shape[1], x.shape[2], x.shape[3])

        if self.mode == "downsample3d":
            if feat_cache[feat_idx] is None:
                feat_cache[feat_idx] = x
                feat_idx += 1
            else:
                x_with_cache = mx.concatenate(
                    [feat_cache[feat_idx][:, -1:, :, :, :], x], axis=1
                )
                feat_cache[feat_idx] = x[:, -1:, :, :, :]
                feat_idx += 1
                x = self.time_conv(x_with_cache, None)

        return x, feat_idx

    def forward_functional(self, x, cache=None):
        b, t, h, w, c = x.shape
        new_cache = None

        if self.mode == "upsample3d":
            if cache is None:
                new_cache = mx.zeros((b, CACHE_T, h, w, c), dtype=x.dtype)
            else:
                cache_input = x
                x = self.time_conv(x, cache)
                new_cache = create_cache_entry(cache_input, cache)
                x = x.reshape(b, t, h, w, 2, c)
                x = x.transpose(0, 1, 4, 2, 3, 5)
                x = x.reshape(b, t * 2, h, w, c)

        t_out = x.shape[1]
        c_out = x.shape[4]
        x = x.reshape(b * t_out, x.shape[2], x.shape[3], c_out)

        if self.mode in ("upsample2d", "upsample3d"):
            x = self.upsample(x)
            x = mx.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)])
            x = mx.conv2d(x, self.conv_weight, stride=1, padding=0)
            x = x + self.conv_bias
        elif self.mode in ("downsample2d", "downsample3d"):
            x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
            x = mx.conv2d(x, self.conv_weight, stride=2, padding=0)
            x = x + self.conv_bias

        x = x.reshape(b, t_out, x.shape[1], x.shape[2], x.shape[3])

        if self.mode == "downsample3d":
            if cache is None:
                new_cache = x
            else:
                x_with_cache = mx.concatenate([cache[:, -1:, :, :, :], x], axis=1)
                new_cache = x[:, -1:, :, :, :]
                x = self.time_conv(x_with_cache, None)

        return x, new_cache


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm1 = RMSNorm(in_dim)
        self.conv1 = CausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = RMSNorm(out_dim)
        self.conv2 = CausalConv3d(out_dim, out_dim, 3, padding=1)
        if in_dim != out_dim:
            self.shortcut = CausalConv3d(in_dim, out_dim, 1)
        else:
            self.shortcut = None

    def __call__(self, x, feat_cache, feat_idx):
        if self.shortcut is not None:
            h = self.shortcut(x)
        else:
            h = x

        residual = self.norm1(x)
        residual = nn.silu(residual)
        cache_input = residual
        residual = self.conv1(residual, feat_cache[feat_idx])
        write_cache(feat_cache, feat_idx, cache_input)
        feat_idx += 1

        residual = self.norm2(residual)
        residual = nn.silu(residual)
        cache_input = residual
        residual = self.conv2(residual, feat_cache[feat_idx])
        write_cache(feat_cache, feat_idx, cache_input)
        feat_idx += 1

        return h + residual, feat_idx

    def forward_functional(self, x, cache1, cache2):
        if self.shortcut is not None:
            h = self.shortcut(x)
        else:
            h = x

        residual = self.norm1(x)
        residual = nn.silu(residual)
        cache_input = residual
        residual = self.conv1(residual, cache1)
        new_cache1 = create_cache_entry(cache_input, cache1)

        residual = self.norm2(residual)
        residual = nn.silu(residual)
        cache_input = residual
        residual = self.conv2(residual, cache2)
        new_cache2 = create_cache_entry(cache_input, cache2)

        return h + residual, new_cache1, new_cache2


class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = RMSNorm(dim)
        scale = 1.0 / dim**0.5
        self.to_qkv_weight = mx.random.uniform(
            low=-scale, high=scale, shape=(dim * 3, 1, 1, dim)
        )
        self.to_qkv_bias = mx.zeros((dim * 3,))
        self.proj_weight = mx.zeros((dim, 1, 1, dim))
        self.proj_bias = mx.zeros((dim,))

    def __call__(self, x):
        identity = x
        b, t, h, w, c = x.shape
        x = x.reshape(b * t, h, w, c)
        x = self.norm(x)
        qkv = mx.conv2d(x, self.to_qkv_weight, stride=1, padding=0) + self.to_qkv_bias
        qkv = qkv.reshape(b * t, h * w, 3, c)
        q, k, v = qkv[:, :, 0, :], qkv[:, :, 1, :], qkv[:, :, 2, :]
        q = q.reshape(b * t, 1, h * w, c)
        k = k.reshape(b * t, 1, h * w, c)
        v = v.reshape(b * t, 1, h * w, c)
        scale = c**-0.5
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        attn = attn.squeeze(1).reshape(b * t, h, w, c)
        out = mx.conv2d(attn, self.proj_weight, stride=1, padding=0) + self.proj_bias
        out = out.reshape(b, t, h, w, c)
        return out + identity
