# MLX implementation of vae2_1.py
import logging
from typing import Optional, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx.utils import tree_unflatten

__all__ = [
    'Wan2_1_VAE',
]

CACHE_T = 2

debug_line = 0


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolution for MLX.
    Expects input in BTHWC format (batch, time, height, width, channels).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Padding order: (W, W, H, H, T, 0)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def __call__(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            x = mx.concatenate([cache_x, x], axis=1)  # Concat along time axis
            padding[4] -= cache_x.shape[1]
        
        # Pad in BTHWC format
        pad_width = [(0, 0), (padding[4], padding[5]), (padding[2], padding[3]), 
                     (padding[0], padding[1]), (0, 0)]
        x = mx.pad(x, pad_width)
        
        result = super().__call__(x)
        return result


class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=False, images=True, bias=False):
        super().__init__()
        self.channel_first = channel_first
        self.images = images
        self.scale = dim**0.5

        # Just keep as 1D - let broadcasting do its magic
        self.gamma = mx.ones((dim,))
        self.bias = mx.zeros((dim,)) if bias else 0.

    def __call__(self, x):
        norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True) + 1e-6)
        x = x / norm
        return x * self.scale * self.gamma + self.bias


class Upsample(nn.Module):
    """
    Upsampling layer that matches PyTorch's behavior.
    """
    def __init__(self, scale_factor, mode='nearest-exact'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode # mode is now unused, but kept for signature consistency

    def __call__(self, x):
        scale_h, scale_w = self.scale_factor
        
        out = mx.repeat(x, int(scale_h), axis=1)  # Repeat along H dimension
        out = mx.repeat(out, int(scale_w), axis=2) # Repeat along W dimension
        
        return out

class AsymmetricPad(nn.Module):
    """A module to apply asymmetric padding, compatible with nn.Sequential."""
    def __init__(self, pad_width: tuple):
        super().__init__()
        self.pad_width = pad_width

    def __call__(self, x):
        return mx.pad(x, self.pad_width)

# Update your Resample class to use 'nearest-exact'
class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1)
            )
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1)
            )
            self.time_conv = CausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        # --- CORRECTED PADDING LOGIC ---
        elif mode == 'downsample2d':
            pad_layer = AsymmetricPad(pad_width=((0, 0), (0, 1), (0, 1), (0, 0)))
            conv_layer = nn.Conv2d(dim, dim, 3, stride=(2, 2), padding=0)
            self.resample = nn.Sequential(pad_layer, conv_layer)

        elif mode == 'downsample3d':
            # The spatial downsampling part uses the same logic
            pad_layer = AsymmetricPad(pad_width=((0, 0), (0, 1), (0, 1), (0, 0)))
            conv_layer = nn.Conv2d(dim, dim, 3, stride=(2, 2), padding=0)
            self.resample = nn.Sequential(pad_layer, conv_layer)
            
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        
        else:
            self.resample = nn.Identity()

    def __call__(self, x, feat_cache=None, feat_idx=[0]):
        # The __call__ method logic remains unchanged from your original code
        b, t, h, w, c = x.shape
        
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, -CACHE_T:, :, :, :]
                    if cache_x.shape[1] < 2 and feat_cache[idx] is not None and feat_cache[idx] != 'Rep':
                        cache_x = mx.concatenate([
                            feat_cache[idx][:, -1:, :, :, :], cache_x
                        ], axis=1)
                    if cache_x.shape[1] < 2 and feat_cache[idx] is not None and feat_cache[idx] == 'Rep':
                        cache_x = mx.concatenate([
                            mx.zeros_like(cache_x), cache_x
                        ], axis=1)
                    
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, t, h, w, 2, c)
                    x = mx.stack([x[:, :, :, :, 0, :], x[:, :, :, :, 1, :]], axis=2)
                    x = x.reshape(b, t * 2, h, w, c)
        
        t = x.shape[1]
        x = x.reshape(b * t, h, w, c)
        
        x = self.resample(x)

        _, h_new, w_new, c_new = x.shape
        x = x.reshape(b, t, h_new, w_new, c_new)

        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, -1:, :, :, :]
                    x = self.time_conv(
                        mx.concatenate([feat_cache[idx][:, -1:, :, :, :], x], axis=1))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        
        return x


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            CausalConv3d(out_dim, out_dim, 3, padding=1)
        )
        
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def __call__(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        
        for i, layer in enumerate(self.residual.layers):
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, -CACHE_T:, :, :, :]
                if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                    cache_x = mx.concatenate([
                        feat_cache[idx][:, -1:, :, :, :], cache_x
                    ], axis=1)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        self.proj.weight = mx.zeros_like(self.proj.weight)

    def __call__(self, x):
        # x is in BTHWC format
        identity = x
        b, t, h, w, c = x.shape
        x = x.reshape(b * t, h, w, c)  # Combine batch and time
        x = self.norm(x)
        # compute query, key, value
        qkv = self.to_qkv(x)  # Output: (b*t, h, w, 3*c)
        qkv = qkv.reshape(b * t, h * w, 3 * c)
        q, k, v = mx.split(qkv, 3, axis=-1)
        
        # Reshape for attention
        q = q.reshape(b * t, h * w, c)
        k = k.reshape(b * t, h * w, c)
        v = v.reshape(b * t, h * w, c)

        # Scaled dot product attention
        scale = 1.0 / mx.sqrt(mx.array(c, dtype=q.dtype))
        scores = (q @ k.transpose(0, 2, 1)) * scale
        weights = mx.softmax(scores, axis=-1)
        x = weights @ v
        x = x.reshape(b * t, h, w, c)

        # output
        x = self.proj(x)
        x = x.reshape(b, t, h, w, c)
        return x + identity


class Encoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0

        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[-1], dims[-1], dropout),
            AttentionBlock(dims[-1]),
            ResidualBlock(dims[-1], dims[-1], dropout)
        )

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(dims[-1], images=False),
            nn.SiLU(),
            CausalConv3d(dims[-1], z_dim, 3, padding=1)
        )

    def __call__(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, -CACHE_T:, :, :, :]
            if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                cache_x = mx.concatenate([
                    feat_cache[idx][:, -1:, :, :, :], cache_x
                ], axis=1)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## downsamples
        for i, layer in enumerate(self.downsamples.layers):
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        for layer in self.middle.layers:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for i, layer in enumerate(self.head.layers):
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, -CACHE_T:, :, :, :]
                if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                    cache_x = mx.concatenate([
                        feat_cache[idx][:, -1:, :, :, :], cache_x
                    ], axis=1)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[False, True, True],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout)
        )

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0

        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(dims[-1], images=False),
            nn.SiLU(),
            CausalConv3d(dims[-1], 3, 3, padding=1)
        )

    def __call__(self, x, feat_cache=None, feat_idx=[0]):
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, -CACHE_T:, :, :, :]
            if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                cache_x = mx.concatenate([
                    feat_cache[idx][:, -1:, :, :, :], cache_x
                ], axis=1)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## middle
        for layer in self.middle.layers:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## upsamples
        for layer in self.upsamples.layers:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for i, layer in enumerate(self.head.layers):
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, -CACHE_T:, :, :, :]
                if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                    cache_x = mx.concatenate([
                        feat_cache[idx][:, -1:, :, :, :], cache_x
                    ], axis=1)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model):
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, CausalConv3d):
            count += 1
    return count


class WanVAE_(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_upsample, dropout)

    def encode(self, x, scale):
        # x is in BTHWC format
        self.clear_cache()
        ## cache
        t = x.shape[1]
        iter_ = 1 + (t - 1) // 4
        ## Split encode input x by time into 1, 4, 4, 4....
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :1, :, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(
                    x[:, 1 + 4 * (i - 1):1 + 4 * i, :, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)
                out = mx.concatenate([out, out_], axis=1)
        
        z = self.conv1(out)
        mu, log_var = mx.split(z, 2, axis=-1)  # Split along channel dimension
        
        if isinstance(scale[0], mx.array):
            # Reshape scale for broadcasting in BTHWC format
            scale_mean = scale[0].reshape(1, 1, 1, 1, self.z_dim)
            scale_std = scale[1].reshape(1, 1, 1, 1, self.z_dim)
            mu = (mu - scale_mean) * scale_std
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()

        return mu, log_var

    def decode(self, z, scale):
        # z is in BTHWC format
        self.clear_cache()
        if isinstance(scale[0], mx.array):
            scale_mean = scale[0].reshape(1, 1, 1, 1, self.z_dim)
            scale_std = scale[1].reshape(1, 1, 1, 1, self.z_dim)
            z = z / scale_std + scale_mean
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[1]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, i:i + 1, :, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)
            else:
                out_ = self.decoder(
                    x[:, i:i + 1, :, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)
                out = mx.concatenate([out, out_], axis=1)
        self.clear_cache()
        return out

    def reparameterize(self, mu, log_var):
        std = mx.exp(0.5 * log_var)
        eps = mx.random.normal(std.shape)
        return eps * std + mu

    def __call__(self, x):
        mu, log_var = self.encode(x, self.scale)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z, self.scale)
        return x_recon, mu, log_var

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs, self.scale)
        if deterministic:
            return mu
        std = mx.exp(0.5 * mx.clip(log_var, -30.0, 20.0))
        return mu + std * mx.random.normal(std.shape)

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        #cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def _video_vae(pretrained_path=None, z_dim=None, **kwargs):
    # params
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0)
    cfg.update(**kwargs)

    # init model
    model = WanVAE_(**cfg)

    # load checkpoint
    if pretrained_path:
        logging.info(f'loading {pretrained_path}')
        weights = mx.load(pretrained_path)
        model.update(tree_unflatten(list(weights.items())))

    return model


class Wan2_1_VAE:

    def __init__(self,
                 z_dim=16,
                 vae_pth='cache/vae_step_411000.pth',
                 dtype=mx.float32):
        self.dtype = dtype

        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = mx.array(mean, dtype=dtype)
        self.std = mx.array(std, dtype=dtype)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = _video_vae(
            pretrained_path=vae_pth,
            z_dim=z_dim,
        )

    def encode(self, videos):
        """
        videos: A list of videos each with shape [C, T, H, W].
        Returns: List of encoded videos in [C, T, H, W] format.
        """
        encoded = []
        for video in videos:
            # Convert CTHW -> BTHWC
            x = mx.expand_dims(video, axis=0)  # Add batch dimension
            x = x.transpose(0, 2, 3, 4, 1)  # BCTHW -> BTHWC
            
            # Encode
            z = self.model.encode(x, self.scale)[0]  # Get mu only
            
            # Convert back BTHWC -> CTHW and remove batch dimension
            z = z.transpose(0, 4, 1, 2, 3)  # BTHWC -> BCTHW
            z = z.squeeze(0)  # Remove batch dimension -> CTHW
            
            encoded.append(z.astype(mx.float32))
        
        return encoded

    def decode(self, zs):
        """
        zs: A list of latent codes each with shape [C, T, H, W].
        Returns: List of decoded videos in [C, T, H, W] format.
        """
        decoded = []
        for z in zs:
            # Convert CTHW -> BTHWC
            x = mx.expand_dims(z, axis=0)  # Add batch dimension
            x = x.transpose(0, 2, 3, 4, 1)  # BCTHW -> BTHWC
           
            # Decode
            x = self.model.decode(x, self.scale)
            
            # Convert back BTHWC -> CTHW and remove batch dimension
            x = x.transpose(0, 4, 1, 2, 3)  # BTHWC -> BCTHW
            x = x.squeeze(0)  # Remove batch dimension -> CTHW
            
            # Clamp values
            x = mx.clip(x, -1, 1)
            
            decoded.append(x.astype(mx.float32))
        
        return decoded