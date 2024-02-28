# Copyright © 2023-2024 Apple Inc.

import glob
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.core import linalg as LA
from mlx.nn.losses import cross_entropy
from mlx.utils import tree_flatten


@dataclass
class CLIPVisionOutput:
    pooler_output: mx.array
    last_hidden_state: mx.array
    hidden_states: Optional[mx.array]


@dataclass
class CLIPTextOutput:
    pooler_output: mx.array
    last_hidden_state: mx.array


@dataclass
class CLIPModelOutput:
    loss: Optional[mx.array]
    text_embeds: Optional[mx.array]
    image_embeds: Optional[mx.array]
    text_model_output: CLIPTextOutput
    vision_model_output: CLIPVisionOutput


@dataclass
class CLIPTextConfig:
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    max_position_embeddings: int
    vocab_size: int
    layer_norm_eps: float


@dataclass
class CLIPVisionConfig:
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_channels: int
    image_size: int
    patch_size: int
    layer_norm_eps: float


@dataclass
class CLIPConfig:
    text_config: CLIPTextConfig
    vision_config: CLIPVisionConfig
    projection_dim: int


def quick_gelu(x: mx.array) -> mx.array:
    """
    A fast GELU approximation https://github.com/hendrycks/GELUs
    """
    return x * mx.sigmoid(1.702 * x)


def clip_loss(logits: mx.array) -> mx.array:
    N, M = logits.shape
    caption_loss = cross_entropy(logits, mx.arange(N), reduction="mean")
    image_loss = cross_entropy(logits.T, mx.arange(M), reduction="mean")
    return (caption_loss + image_loss) / 2.0


class Attention(nn.Module):
    def __init__(
        self,
        dims: int,
        num_heads: int,
        query_input_dims: Optional[int] = None,
        key_input_dims: Optional[int] = None,
        value_input_dims: Optional[int] = None,
        value_dims: Optional[int] = None,
        value_output_dims: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()

        if (dims % num_heads) != 0:
            raise ValueError(
                "The input feature dimensions should be divisible by the "
                f"number of heads ({dims} % {num_heads}) != 0"
            )

        query_input_dims = query_input_dims or dims
        key_input_dims = key_input_dims or dims
        value_input_dims = value_input_dims or key_input_dims
        value_dims = value_dims or dims
        value_output_dims = value_output_dims or dims

        self.num_heads = num_heads
        self.q_proj = nn.Linear(query_input_dims, dims, bias=bias)
        self.k_proj = nn.Linear(key_input_dims, dims, bias=bias)
        self.v_proj = nn.Linear(value_input_dims, value_dims, bias=bias)
        self.out_proj = nn.Linear(value_dims, value_output_dims, bias=bias)

    def __call__(self, queries, keys, values, mask=None):
        queries = self.q_proj(queries)
        keys = self.k_proj(keys)
        values = self.v_proj(values)

        num_heads = self.num_heads
        B, L, D = queries.shape
        _, S, _ = keys.shape
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, num_heads, -1).transpose(0, 2, 3, 1)
        values = values.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)

        scale = math.sqrt(1 / queries.shape[-1])
        scores = (queries * scale) @ keys
        if mask is not None:
            scores = scores + mask.astype(scores.dtype)
        scores = mx.softmax(scores, axis=-1)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.out_proj(values_hat)


class MLP(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config = config
        self.activation_fn = quick_gelu
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    """The transformer encoder layer from CLIP."""

    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        # Add biases to the attention projections
        self.self_attn = Attention(
            config.hidden_size, config.num_attention_heads, bias=True
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        y = self.layer_norm1(x)
        y = self.self_attn(y, y, y, mask)
        x = x + y
        y = self.layer_norm2(x)
        y = self.mlp(y)
        return x + y


class TextEmbeddings(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, embed_dim
        )

    def __call__(self, x: mx.array) -> mx.array:
        embeddings = self.token_embedding(x)
        embeddings += self.position_embedding.weight[: x.shape[1]]
        return embeddings


class Encoder(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]


class ClipTextModel(nn.Module):
    """Implements the text encoder transformer from CLIP."""

    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.embeddings = TextEmbeddings(config)
        self.encoder = Encoder(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

    def __call__(self, x: mx.array) -> CLIPTextOutput:
        B, N = x.shape
        eot_tokens = mx.argmax(x, axis=-1)
        x = self.embeddings(x)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(N, x.dtype)
        for l in self.encoder.layers:
            x = l(x, mask)
        last_hidden_state = self.final_layer_norm(x)
        pooler_output = last_hidden_state[mx.arange(B), eot_tokens]

        return CLIPTextOutput(
            pooler_output=pooler_output, last_hidden_state=last_hidden_state
        )


class VisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = mx.zeros((config.hidden_size,))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        batch_size = x.shape[0]
        # Patchify using conv:
        # [batch_size, sqrt(num_patches), sqrt(num_patches), embed_dim]
        patch_embeddings = self.patch_embedding(x)
        # [batch_size, num_patches, embed_dim]
        patch_embeddings = mx.flatten(patch_embeddings, start_axis=1, end_axis=2)
        embed_dim = patch_embeddings.shape[-1]
        # Prepend <CLS> embeddings
        # [batch_size, 1, embed_dim]
        cls_embeddings = mx.broadcast_to(
            self.class_embedding, (batch_size, 1, embed_dim)
        )
        # [batch_size, num_patches + 1, embed_dim]
        embeddings = mx.concatenate((cls_embeddings, patch_embeddings), axis=1)
        # Add positional encoding
        embeddings += self.position_embedding.weight
        return embeddings


class ClipVisionModel(nn.Module):
    """Implements the vision encoder transformer from CLIP."""

    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.embeddings = VisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size)
        self.encoder = Encoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size)

    def __call__(
        self,
        x: mx.array,
        output_hidden_states: Optional[bool] = None,
    ) -> CLIPVisionOutput:
        x = self.embeddings(x)
        x = self.pre_layrnorm(x)

        encoder_states = (x,) if output_hidden_states else None

        for l in self.encoder.layers:
            x = l(x, mask=None)
            if output_hidden_states:
                encoder_states = encoder_states + (x,)

        # Extract <CLS> token embedding
        pooler_output = self.post_layernorm(x[:, 0, :])
        return CLIPVisionOutput(
            pooler_output=pooler_output,
            last_hidden_state=x,
            hidden_states=encoder_states,
        )


class CLIPModel(nn.Module):
    def __init__(self, config: CLIPConfig):
        self.text_model = ClipTextModel(config.text_config)
        self.vision_model = ClipVisionModel(config.vision_config)

        text_embed_dim = config.text_config.hidden_size
        vision_embed_dim = config.vision_config.hidden_size
        projection_dim = config.projection_dim

        self.visual_projection = nn.Linear(vision_embed_dim, projection_dim, bias=False)
        self.text_projection = nn.Linear(text_embed_dim, projection_dim, bias=False)
        self.logit_scale = mx.array(0.0)

    def get_text_features(self, x: mx.array) -> mx.array:
        return self.text_projection(self.text_model(x).pooler_output)

    def get_image_features(self, x: mx.array) -> mx.array:
        return self.visual_projection(self.vision_model(x).pooler_output)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        return_loss=False,
    ) -> CLIPModelOutput:
        if input_ids is not None:
            text_model_output = self.text_model(input_ids)
            text_embeds = self.text_projection(text_model_output.pooler_output)
            text_embeds = text_embeds / LA.norm(text_embeds, axis=-1, keepdims=True)
        else:
            text_embeds = None
            text_model_output = None

        if pixel_values is not None:
            vision_model_output = self.vision_model(pixel_values)
            image_embeds = self.visual_projection(vision_model_output.pooler_output)
            image_embeds = image_embeds / LA.norm(image_embeds, axis=-1, keepdims=True)
        else:
            image_embeds = None
            vision_model_output = None

        if return_loss and (input_ids is None or pixel_values is None):
            raise ValueError("Must provide text and image inputs to compute loss.")

        if return_loss:
            logit_scale = mx.exp(self.logit_scale)
            logits = (text_embeds @ image_embeds.T) * logit_scale
            loss = clip_loss(logits)
        else:
            loss = None

        return CLIPModelOutput(
            loss=loss,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            vision_model_output=vision_model_output,
            text_model_output=text_model_output,
        )

    @staticmethod
    def from_pretrained(path: str):
        path = Path(path)

        with open(path / "config.json", "r") as fid:
            config = json.load(fid)

        text_config = config["text_config"]
        text_config = CLIPTextConfig(
            num_hidden_layers=text_config["num_hidden_layers"],
            hidden_size=text_config["hidden_size"],
            intermediate_size=text_config["intermediate_size"],
            num_attention_heads=text_config["num_attention_heads"],
            max_position_embeddings=text_config["max_position_embeddings"],
            vocab_size=text_config["vocab_size"],
            layer_norm_eps=text_config["layer_norm_eps"],
        )

        vision_config = config["vision_config"]

        vision_config = CLIPVisionConfig(
            num_hidden_layers=vision_config["num_hidden_layers"],
            hidden_size=vision_config["hidden_size"],
            intermediate_size=vision_config["intermediate_size"],
            num_attention_heads=vision_config["num_attention_heads"],
            num_channels=3,
            image_size=vision_config["image_size"],
            patch_size=vision_config["patch_size"],
            layer_norm_eps=vision_config["layer_norm_eps"],
        )

        config = CLIPConfig(
            text_config=text_config,
            vision_config=vision_config,
            projection_dim=config["projection_dim"],
        )
        model = CLIPModel(config)
        weight_files = glob.glob(str(path / "*.safetensors"))
        if not weight_files:
            logging.error(f"No safetensors found in {path}")
            raise FileNotFoundError(f"No safetensors found in {path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        weights = model.sanitize(weights)
        model.load_weights(list(weights.items()))
        return model

    @staticmethod
    def sanitize(weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            elif "patch_embedding.weight" in k:
                # pytorch conv2d expects the weight tensor to be of shape [out_channels, in_channels, kH, KW]
                # mlx conv2d expects the weight tensor to be of shape [out_channels, kH, KW, in_channels]
                sanitized_weights[k] = v.transpose(0, 2, 3, 1)
            else:
                sanitized_weights[k] = v

        return sanitized_weights
