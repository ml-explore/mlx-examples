# Copyright © 2023 Apple Inc.

from dataclasses import dataclass


@dataclass
class CLIPTextConfig:
    num_hidden_layers: int = 12
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_attention_heads: int = 8
    max_position_embeddings: int = 77
    vocab_size: int = 49408
    initializer_factor: float = 1.0


@dataclass
class CLIPVisionConfig:
    num_hidden_layers: int = 12
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 32
    initializer_range: float = 0.02
    initializer_factor: float = 1.0


@dataclass
class CLIPConfig:
    text_config: CLIPTextConfig
    vision_config: CLIPVisionConfig
    projection_dim: int = 512
    initializer_factor: float = 1.0
    logit_scale_init_value: float = 2.6592