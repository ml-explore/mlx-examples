# Copyright © 2023 Apple Inc.

import math
from typing import List

import mlx.core as mx
import mlx.nn as nn

from .config import AutoencoderConfig
from .unet import ResnetBlock2D, upsample_nearest


class Attention(nn.Module):
    """A single head unmasked attention for use with the VAE."""

    def __init__(self, dims: int, norm_groups: int = 32):
        super().__init__()

        self.group_norm = nn.GroupNorm(norm_groups, dims, pytorch_compatible=True)
        self.query_proj = nn.Linear(dims, dims)
        self.key_proj = nn.Linear(dims, dims)
        self.value_proj = nn.Linear(dims, dims)
        self.out_proj = nn.Linear(dims, dims)

    def __call__(self, x):
        B, H, W, C = x.shape

        y = self.group_norm(x)

        queries = self.query_proj(y).reshape(B, H * W, C)
        keys = self.key_proj(y).reshape(B, H * W, C)
        values = self.value_proj(y).reshape(B, H * W, C)

        scale = 1 / math.sqrt(queries.shape[-1])
        scores = (queries * scale) @ keys.transpose(0, 2, 1)
        attn = mx.softmax(scores, axis=-1)
        y = (attn @ values).reshape(B, H, W, C)

        y = self.out_proj(y)
        x = x + y

        return x


class EncoderDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        resnet_groups: int = 32,
        add_downsample=True,
        add_upsample=True,
    ):
        super().__init__()

        # Add the resnet blocks
        self.resnets = [
            ResnetBlock2D(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                groups=resnet_groups,
            )
            for i in range(num_layers)
        ]

        # Add an optional downsampling layer
        if add_downsample:
            self.downsample = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=0
            )

        # or upsampling layer
        if add_upsample:
            self.upsample = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1
            )

    def __call__(self, x):
        for resnet in self.resnets:
            x = resnet(x)

        if "downsample" in self:
            x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
            x = self.downsample(x)

        if "upsample" in self:
            x = self.upsample(upsample_nearest(x))

        return x


class Encoder(nn.Module):
    """Implements the encoder side of the Autoencoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_out_channels: List[int] = [64],
        layers_per_block: int = 2,
        resnet_groups: int = 32,
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1
        )

        channels = [block_out_channels[0]] + list(block_out_channels)
        self.down_blocks = [
            EncoderDecoderBlock2D(
                in_channels,
                out_channels,
                num_layers=layers_per_block,
                resnet_groups=resnet_groups,
                add_downsample=i < len(block_out_channels) - 1,
                add_upsample=False,
            )
            for i, (in_channels, out_channels) in enumerate(zip(channels, channels[1:]))
        ]

        self.mid_blocks = [
            ResnetBlock2D(
                in_channels=block_out_channels[-1],
                out_channels=block_out_channels[-1],
                groups=resnet_groups,
            ),
            Attention(block_out_channels[-1], resnet_groups),
            ResnetBlock2D(
                in_channels=block_out_channels[-1],
                out_channels=block_out_channels[-1],
                groups=resnet_groups,
            ),
        ]

        self.conv_norm_out = nn.GroupNorm(
            resnet_groups, block_out_channels[-1], pytorch_compatible=True
        )
        self.conv_out = nn.Conv2d(block_out_channels[-1], out_channels, 3, padding=1)

    def __call__(self, x):
        x = self.conv_in(x)

        for l in self.down_blocks:
            x = l(x)

        x = self.mid_blocks[0](x)
        x = self.mid_blocks[1](x)
        x = self.mid_blocks[2](x)

        x = self.conv_norm_out(x)
        x = nn.silu(x)
        x = self.conv_out(x)

        return x


class Decoder(nn.Module):
    """Implements the decoder side of the Autoencoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_out_channels: List[int] = [64],
        layers_per_block: int = 2,
        resnet_groups: int = 32,
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1
        )

        self.mid_blocks = [
            ResnetBlock2D(
                in_channels=block_out_channels[-1],
                out_channels=block_out_channels[-1],
                groups=resnet_groups,
            ),
            Attention(block_out_channels[-1], resnet_groups),
            ResnetBlock2D(
                in_channels=block_out_channels[-1],
                out_channels=block_out_channels[-1],
                groups=resnet_groups,
            ),
        ]

        channels = list(reversed(block_out_channels))
        channels = [channels[0]] + channels
        self.up_blocks = [
            EncoderDecoderBlock2D(
                in_channels,
                out_channels,
                num_layers=layers_per_block,
                resnet_groups=resnet_groups,
                add_downsample=False,
                add_upsample=i < len(block_out_channels) - 1,
            )
            for i, (in_channels, out_channels) in enumerate(zip(channels, channels[1:]))
        ]

        self.conv_norm_out = nn.GroupNorm(
            resnet_groups, block_out_channels[0], pytorch_compatible=True
        )
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def __call__(self, x):
        x = self.conv_in(x)

        x = self.mid_blocks[0](x)
        x = self.mid_blocks[1](x)
        x = self.mid_blocks[2](x)

        for l in self.up_blocks:
            x = l(x)

        x = self.conv_norm_out(x)
        x = nn.silu(x)
        x = self.conv_out(x)

        return x


class Autoencoder(nn.Module):
    """The autoencoder that allows us to perform diffusion in the latent space."""

    def __init__(self, config: AutoencoderConfig):
        super().__init__()

        self.latent_channels = config.latent_channels_in
        self.scaling_factor = config.scaling_factor
        self.encoder = Encoder(
            config.in_channels,
            config.latent_channels_out,
            config.block_out_channels,
            config.layers_per_block,
            resnet_groups=config.norm_num_groups,
        )
        self.decoder = Decoder(
            config.latent_channels_in,
            config.out_channels,
            config.block_out_channels,
            config.layers_per_block + 1,
            resnet_groups=config.norm_num_groups,
        )

        self.quant_proj = nn.Linear(
            config.latent_channels_out, config.latent_channels_out
        )
        self.post_quant_proj = nn.Linear(
            config.latent_channels_in, config.latent_channels_in
        )

    def decode(self, z):
        z = z / self.scaling_factor
        return self.decoder(self.post_quant_proj(z))

    def encode(self, x):
        x = self.encoder(x)
        x = self.quant_proj(x)
        mean, logvar = x.split(2, axis=-1)
        mean = mean * self.scaling_factor
        logvar = logvar + 2 * math.log(self.scaling_factor)

        return mean, logvar

    def __call__(self, x, key=None):
        mean, logvar = self.encode(x)
        z = mx.random.normal(mean.shape, key=key) * mx.exp(0.5 * logvar) + mean
        x_hat = self.decode(z)

        return dict(x_hat=x_hat, z=z, mean=mean, logvar=logvar)
