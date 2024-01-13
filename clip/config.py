# Copyright © 2023 Apple Inc.

from dataclasses import dataclass


@dataclass
class CLIPTextConfig:
    num_hidden_layers: int = 12,
    hidden_size: int = 768,
    intermediate_size: int = 3072,
    projection_dim: int = 768
    num_attention_heads: int = 12,
    max_position_embeddings: int = 77,
    vocab_size: int = 49408


@dataclass
class CLIPVisionConfig:
    hidden_size: int = 768,
    intermediate_size: int = 3072,
    projection_dim: int = 512,
    num_hidden_layers: int = 12,
    num_attention_heads: int = 12,
    num_channels: int = 3,
    image_size: int = 224,
    patch_size: int = 32,
    layer_norm_eps: int = 1e-5,
    attention_dropout = 0.0
