# Copyright © 2023 Apple Inc.

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import UNetConfig


def upsample_nearest(x, scale: int = 2):
    B, H, W, C = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, scale, W, scale, C))
    x = x.reshape(B, H * scale, W * scale, C)

    return x


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def __call__(self, x):
        x = self.linear_1(x)
        x = nn.silu(x)
        x = self.linear_2(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        model_dims: int,
        num_heads: int,
        hidden_dims: Optional[int] = None,
        memory_dims: Optional[int] = None,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(model_dims)
        self.attn1 = nn.MultiHeadAttention(model_dims, num_heads)
        self.attn1.out_proj.bias = mx.zeros(model_dims)

        memory_dims = memory_dims or model_dims
        self.norm2 = nn.LayerNorm(model_dims)
        self.attn2 = nn.MultiHeadAttention(
            model_dims, num_heads, key_input_dims=memory_dims
        )
        self.attn2.out_proj.bias = mx.zeros(model_dims)

        hidden_dims = hidden_dims or 4 * model_dims
        self.norm3 = nn.LayerNorm(model_dims)
        self.linear1 = nn.Linear(model_dims, hidden_dims)
        self.linear2 = nn.Linear(model_dims, hidden_dims)
        self.linear3 = nn.Linear(hidden_dims, model_dims)

    def __call__(self, x, memory, attn_mask, memory_mask):
        # Self attention
        y = self.norm1(x)
        y = self.attn1(y, y, y, attn_mask)
        x = x + y

        # Cross attention
        y = self.norm2(x)
        y = self.attn2(y, memory, memory, memory_mask)
        x = x + y

        # FFN
        y = self.norm3(x)
        y_a = self.linear1(y)
        y_b = self.linear2(y)
        y = y_a * nn.gelu_approx(y_b)  # approximate gelu?
        y = self.linear3(y)
        x = x + y

        return x


class Transformer2D(nn.Module):
    """A transformer model for inputs with 2 spatial dimensions."""

    def __init__(
        self,
        in_channels: int,
        model_dims: int,
        encoder_dims: int,
        num_heads: int,
        num_layers: int = 1,
        norm_num_groups: int = 32,
    ):
        super().__init__()

        self.norm = nn.GroupNorm(norm_num_groups, in_channels, pytorch_compatible=True)
        self.proj_in = nn.Linear(in_channels, model_dims)
        self.transformer_blocks = [
            TransformerBlock(model_dims, num_heads, memory_dims=encoder_dims)
            for i in range(num_layers)
        ]
        self.proj_out = nn.Linear(model_dims, in_channels)

    def __call__(self, x, encoder_x, attn_mask, encoder_attn_mask):
        # Save the input to add to the output
        input_x = x

        # Perform the input norm and projection
        B, H, W, C = x.shape
        x = self.norm(x).reshape(B, -1, C)
        x = self.proj_in(x)

        # Apply the transformer
        for block in self.transformer_blocks:
            x = block(x, encoder_x, attn_mask, encoder_attn_mask)

        # Apply the output projection and reshape
        x = self.proj_out(x)
        x = x.reshape(B, H, W, C)

        return x + input_x


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        groups: int = 32,
        temb_channels: Optional[int] = None,
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        self.norm1 = nn.GroupNorm(groups, in_channels, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        if in_channels != out_channels:
            self.conv_shortcut = nn.Linear(in_channels, out_channels)

    def __call__(self, x, temb=None):
        if temb is not None:
            temb = self.time_emb_proj(nn.silu(temb))

        y = self.norm1(x)
        y = nn.silu(y)
        y = self.conv1(y)
        if temb is not None:
            y = y + temb[:, None, None, :]
        y = self.norm2(y)
        y = nn.silu(y)
        y = self.conv2(y)

        x = y + (x if "conv_shortcut" not in self else self.conv_shortcut(x))

        return x


class UNetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        prev_out_channels: Optional[int] = None,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        num_attention_heads: int = 8,
        cross_attention_dim=1280,
        resnet_groups: int = 32,
        add_downsample=True,
        add_upsample=True,
        add_cross_attention=True,
    ):
        super().__init__()

        # Prepare the in channels list for the resnets
        if prev_out_channels is None:
            in_channels_list = [in_channels] + [out_channels] * (num_layers - 1)
        else:
            in_channels_list = [prev_out_channels] + [out_channels] * (num_layers - 1)
            res_channels_list = [out_channels] * (num_layers - 1) + [in_channels]
            in_channels_list = [
                a + b for a, b in zip(in_channels_list, res_channels_list)
            ]

        # Add resnet blocks that also process the time embedding
        self.resnets = [
            ResnetBlock2D(
                in_channels=ic,
                out_channels=out_channels,
                temb_channels=temb_channels,
                groups=resnet_groups,
            )
            for ic in in_channels_list
        ]

        # Add optional cross attention layers
        if add_cross_attention:
            self.attentions = [
                Transformer2D(
                    in_channels=out_channels,
                    model_dims=out_channels,
                    num_heads=num_attention_heads,
                    num_layers=transformer_layers_per_block,
                    encoder_dims=cross_attention_dim,
                )
                for i in range(num_layers)
            ]

        # Add an optional downsampling layer
        if add_downsample:
            self.downsample = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1
            )

        # or upsampling layer
        if add_upsample:
            self.upsample = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1
            )

    def __call__(
        self,
        x,
        encoder_x=None,
        temb=None,
        attn_mask=None,
        encoder_attn_mask=None,
        residual_hidden_states=None,
    ):
        output_states = []

        for i in range(len(self.resnets)):
            if residual_hidden_states is not None:
                x = mx.concatenate([x, residual_hidden_states.pop()], axis=-1)

            x = self.resnets[i](x, temb)

            if "attentions" in self:
                x = self.attentions[i](x, encoder_x, attn_mask, encoder_attn_mask)

            output_states.append(x)

        if "downsample" in self:
            x = self.downsample(x)
            output_states.append(x)

        if "upsample" in self:
            x = self.upsample(upsample_nearest(x))
            output_states.append(x)

        return x, output_states


class UNetModel(nn.Module):
    """The conditional 2D UNet model that actually performs the denoising."""

    def __init__(self, config: UNetConfig):
        super().__init__()

        self.conv_in = nn.Conv2d(
            config.in_channels,
            config.block_out_channels[0],
            config.conv_in_kernel,
            padding=(config.conv_in_kernel - 1) // 2,
        )

        self.timesteps = nn.SinusoidalPositionalEncoding(
            config.block_out_channels[0],
            max_freq=1,
            min_freq=math.exp(
                -math.log(10000) + 2 * math.log(10000) / config.block_out_channels[0]
            ),
            scale=1.0,
            cos_first=True,
            full_turns=False,
        )
        self.time_embedding = TimestepEmbedding(
            config.block_out_channels[0],
            config.block_out_channels[0] * 4,
        )

        # Make the downsampling blocks
        block_channels = [config.block_out_channels[0]] + list(
            config.block_out_channels
        )
        self.down_blocks = [
            UNetBlock2D(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=config.block_out_channels[0] * 4,
                num_layers=config.layers_per_block[i],
                transformer_layers_per_block=config.transformer_layers_per_block[i],
                num_attention_heads=config.num_attention_heads[i],
                cross_attention_dim=config.cross_attention_dim[i],
                resnet_groups=config.norm_num_groups,
                add_downsample=(i < len(config.block_out_channels) - 1),
                add_upsample=False,
                add_cross_attention=(i < len(config.block_out_channels) - 1),
            )
            for i, (in_channels, out_channels) in enumerate(
                zip(block_channels, block_channels[1:])
            )
        ]

        # Make the middle block
        self.mid_blocks = [
            ResnetBlock2D(
                in_channels=config.block_out_channels[-1],
                out_channels=config.block_out_channels[-1],
                temb_channels=config.block_out_channels[0] * 4,
                groups=config.norm_num_groups,
            ),
            Transformer2D(
                in_channels=config.block_out_channels[-1],
                model_dims=config.block_out_channels[-1],
                num_heads=config.num_attention_heads[-1],
                num_layers=config.transformer_layers_per_block[-1],
                encoder_dims=config.cross_attention_dim[-1],
            ),
            ResnetBlock2D(
                in_channels=config.block_out_channels[-1],
                out_channels=config.block_out_channels[-1],
                temb_channels=config.block_out_channels[0] * 4,
                groups=config.norm_num_groups,
            ),
        ]

        # Make the upsampling blocks
        block_channels = (
            [config.block_out_channels[0]]
            + list(config.block_out_channels)
            + [config.block_out_channels[-1]]
        )
        self.up_blocks = [
            UNetBlock2D(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=config.block_out_channels[0] * 4,
                prev_out_channels=prev_out_channels,
                num_layers=config.layers_per_block[i] + 1,
                transformer_layers_per_block=config.transformer_layers_per_block[i],
                num_attention_heads=config.num_attention_heads[i],
                cross_attention_dim=config.cross_attention_dim[i],
                resnet_groups=config.norm_num_groups,
                add_downsample=False,
                add_upsample=(i > 0),
                add_cross_attention=(i < len(config.block_out_channels) - 1),
            )
            for i, (in_channels, out_channels, prev_out_channels) in reversed(
                list(
                    enumerate(
                        zip(block_channels, block_channels[1:], block_channels[2:])
                    )
                )
            )
        ]

        self.conv_norm_out = nn.GroupNorm(
            config.norm_num_groups,
            config.block_out_channels[0],
            pytorch_compatible=True,
        )
        self.conv_out = nn.Conv2d(
            config.block_out_channels[0],
            config.out_channels,
            config.conv_out_kernel,
            padding=(config.conv_out_kernel - 1) // 2,
        )

    def __call__(self, x, timestep, encoder_x, attn_mask=None, encoder_attn_mask=None):
        # Compute the time embeddings
        temb = self.timesteps(timestep).astype(x.dtype)
        temb = self.time_embedding(temb)

        # Preprocess the input
        x = self.conv_in(x)

        # Run the downsampling part of the unet
        residuals = [x]
        for block in self.down_blocks:
            x, res = block(
                x,
                encoder_x=encoder_x,
                temb=temb,
                attn_mask=attn_mask,
                encoder_attn_mask=encoder_attn_mask,
            )
            residuals.extend(res)

        # Run the middle part of the unet
        x = self.mid_blocks[0](x, temb)
        x = self.mid_blocks[1](x, encoder_x, attn_mask, encoder_attn_mask)
        x = self.mid_blocks[2](x, temb)

        # Run the upsampling part of the unet
        for block in self.up_blocks:
            x, _ = block(
                x,
                encoder_x=encoder_x,
                temb=temb,
                attn_mask=attn_mask,
                encoder_attn_mask=encoder_attn_mask,
                residual_hidden_states=residuals,
            )

        # Postprocess the output
        x = self.conv_norm_out(x)
        x = nn.silu(x)
        x = self.conv_out(x)

        return x
