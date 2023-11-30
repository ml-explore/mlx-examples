# Copyright © 2023 Apple Inc.

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class AutoencoderConfig:
    in_channels: int = 3
    out_channels: int = 3
    latent_channels_out: int = 8
    latent_channels_in: int = 4
    block_out_channels: Tuple[int] = (128, 256, 512, 512)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    scaling_factor: float = 0.18215


@dataclass
class CLIPTextModelConfig:
    num_layers: int = 23
    model_dims: int = 1024
    num_heads: int = 16
    max_length: int = 77
    vocab_size: int = 49408


@dataclass
class UNetConfig:
    in_channels: int = 4
    out_channels: int = 4
    conv_in_kernel: int = 3
    conv_out_kernel: int = 3
    block_out_channels: Tuple[int] = (320, 640, 1280, 1280)
    layers_per_block: Tuple[int] = (2, 2, 2, 2)
    mid_block_layers: int = 2
    transformer_layers_per_block: Tuple[int] = (1, 1, 1, 1)
    num_attention_heads: Tuple[int] = (5, 10, 20, 20)
    cross_attention_dim: Tuple[int] = (1024,) * 4
    norm_num_groups: int = 32


@dataclass
class DiffusionConfig:
    beta_schedule: str = "scaled_linear"
    beta_start: float = 0.00085
    beta_end: float = 0.012
    num_train_steps: int = 1000
