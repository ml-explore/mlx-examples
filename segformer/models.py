import glob
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn


@dataclass
class ModelArgs:
    num_channels: int
    num_encoder_blocks: int
    depths: list[int]
    sr_ratios: list[int]
    hidden_sizes: list[int]
    patch_sizes: list[int]
    strides: list[int]
    num_attention_heads: list[int]
    mlp_ratios: list[int]
    hidden_act: str
    layer_norm_eps: float
    decoder_hidden_size: int

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class SegformerOverlapPatchEmbeddings(nn.Module):
    """Overlap Patch Embeddings"""

    def __init__(
        self, patch_size: int, stride: int, num_channels: int, hidden_size: int
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=num_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, x: mx.array) -> mx.array:
        x = self.proj(x)
        (b, h, w, c) = x.shape
        x = x.flatten(1, 2)
        x = self.layer_norm(x)
        x = x.reshape(b, h, w, c)
        return x


class SegformerEfficientSelfAttention(nn.Module):
    """Efficient Self Attention"""

    def __init__(
        self, hidden_size: int, num_attention_heads: int, sequence_reduction_ratio: int
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size must be divisible by the number of attention heads."
            )
        self.num_attention_heads = num_attention_heads
        attention_head_size: int = hidden_size // num_attention_heads
        self.attention_head_size = attention_head_size
        all_head_size: int = attention_head_size * num_attention_heads
        self.query = nn.Linear(hidden_size, all_head_size)
        self.key = nn.Linear(hidden_size, all_head_size)
        self.value = nn.Linear(hidden_size, all_head_size)
        if sequence_reduction_ratio > 1:
            self.sr = nn.Conv2d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=sequence_reduction_ratio,
                stride=sequence_reduction_ratio,
            )
            self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        else:
            self.sr = None
            self.layer_norm = None

    def _transpose_for_scores(self, x: mx.array) -> mx.array:
        (b, n, _) = x.shape
        x = x.reshape(b, n, self.num_attention_heads, self.attention_head_size)
        x = x.transpose(0, 2, 1, 3)
        return x

    def forward(self, x: mx.array) -> mx.array:
        hidden_states = x.flatten(1, 2)
        query = self.query(hidden_states)
        query = self._transpose_for_scores(query)
        if self.sr is not None and self.layer_norm is not None:
            value = self.sr(x)
            value = value.flatten(1, 2)
            value = self.layer_norm(value)
            hidden_states = value

        key = self.key(hidden_states)
        key = self._transpose_for_scores(key)

        value = self.value(hidden_states)
        value = self._transpose_for_scores(value)

        attention_scores = mx.matmul(query, key.transpose(0, 1, 3, 2))
        attention_scores = attention_scores / (self.attention_head_size**0.5)
        attention_probs = mx.softmax(attention_scores, axis=-1)
        hidden_states = mx.matmul(attention_probs, value)
        hidden_states = hidden_states.transpose(0, 2, 1, 3).flatten(1, 2)
        return hidden_states


class SegformerSelfOutput(nn.Module):
    """Self Output"""

    def __init__(self, hidden_size: int):
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: mx.array) -> mx.array:
        return self.dense(x)


class SegformerAttention(nn.Module):
    """Segformer Attention"""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        sequence_reduction_ratio: int,
    ):
        super().__init__()
        self.self = SegformerEfficientSelfAttention(
            hidden_size, num_attention_heads, sequence_reduction_ratio
        )
        self.output = SegformerSelfOutput(hidden_size)

    def forward(self, x: mx.array) -> mx.array:
        x = self.self(x)
        x = self.output(x)
        return x


class SegformerDWConv(nn.Module):
    """Depthwise Convolution"""

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=1,
            groups=dim,
            padding=1,
        )

    def forward(self, x: mx.array) -> mx.array:
        x = self.dwconv(x)
        return x


class SegformerMixFFN(nn.Module):
    """Mix Feed Forward Network"""

    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.dense_1 = nn.Linear(in_features, hidden_features)
        self.dwconv = SegformerDWConv(hidden_features)
        self.act = nn.GELU()
        self.dense_2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: mx.array) -> mx.array:
        (b, _, h, w) = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.dense_1(x)
        (_, c, _) = x.shape
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.dwconv(x)
        x = self.act(x)
        x = x.flatten(2).transpose(1, 2)
        (_, c, _) = x.shape
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x


class SegformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        sequence_reduction_ratio: int,
        mlp_ratio: int,
    ):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attention = SegformerAttention(
            hidden_size, num_attention_heads, sequence_reduction_ratio
        )
        self.layer_norm_2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = SegformerMixFFN(hidden_size, hidden_size * mlp_ratio, hidden_size)

    def forward(self, x: mx.array) -> mx.array:
        shape = x.shape
        hidden_states = x.flatten(2).transpose(1, 2)
        layer_norm_output = self.layer_norm_1(hidden_states)
        layer_norm_output = layer_norm_output.transpose(1, 2).reshape(shape)
        attention_output = self.attention(attention_output)
        hidden_states = attention_output + hidden_states

        layer_norm_output = self.layer_norm_2(hidden_states)
        layer_norm_output = layer_norm_output.transpose(1, 2).reshape(shape)
        mlp_output = self.mlp(layer_norm_output)
        hidden_states = hidden_states.transpose(0, 2, 1, 3).flatten(2)

        return mlp_output + hidden_states


class SegformerEncoder(nn.Module):
    pass


class SegformerClassificationModel(nn.Module):
    def __init__(self, config: ModelArgs, num_classes: int):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.encoder = SegformerEncoder(config)
        # self.decoder = SegformerDecoder(config, num_classes)
