# Copyright © 2023-2024 Apple Inc.

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.core import linalg as LA
from mlx.nn.losses import cross_entropy
from mlx.utils import tree_flatten


@dataclass
class CLIPVisionOutput:
    pooler_output: mx.array
    last_hidden_state: mx.array


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


@dataclass
class CLIPVisionConfig:
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_channels: int
    image_size: int
    patch_size: int


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


class CLIPEncoderLayer(nn.TransformerEncoderLayer):
    """The transformer encoder layer from CLIP."""

    def __init__(self, hidden_dim: int, intermediate_dim: int, num_heads: int):
        super().__init__(
            dims=hidden_dim,
            mlp_dims=intermediate_dim,
            num_heads=num_heads,
            activation=quick_gelu,
            norm_first=True,
        )
        # Add biases to the attention projections
        self.attention = nn.MultiHeadAttention(hidden_dim, num_heads, bias=True)


class CLIPTextModel(nn.Module):
    """Implements the text encoder transformer from CLIP."""

    def __init__(self, config: CLIPTextConfig):
        super().__init__()

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = mx.zeros(
            (config.max_position_embeddings, config.hidden_size)
        )
        self.layers = [
            CLIPEncoderLayer(
                config.hidden_size, config.intermediate_size, config.num_attention_heads
            )
            for _ in range(config.num_hidden_layers)
        ]
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

    def _embed(self, x: mx.array) -> mx.array:
        embeddings = self.token_embedding(x)
        embeddings += self.position_embedding[: x.shape[1]]
        return embeddings

    def __call__(self, x: mx.array) -> CLIPTextOutput:
        B, N = x.shape
        eot_tokens = mx.argmax(x, axis=-1)
        x = self._embed(x)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(N, x.dtype)
        for l in self.layers:
            x = l(x, mask)
        last_hidden_state = self.final_layer_norm(x)
        pooler_output = last_hidden_state[mx.arange(B), eot_tokens]

        return CLIPTextOutput(
            pooler_output=pooler_output, last_hidden_state=last_hidden_state
        )


class CLIPVisionModel(nn.Module):
    """Implements the vision encoder transformer from CLIP."""

    def __init__(self, config: CLIPVisionConfig):
        super().__init__()

        self.class_embedding = mx.zeros((config.hidden_size,))
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        num_patches = (config.image_size // config.patch_size) ** 2
        num_positions = num_patches + 1
        self.position_embedding = mx.zeros((num_positions, config.hidden_size))
        self.pre_layernorm = nn.LayerNorm(config.hidden_size)
        self.layers = [
            CLIPEncoderLayer(
                config.hidden_size, config.intermediate_size, config.num_attention_heads
            )
            for _ in range(config.num_hidden_layers)
        ]
        self.post_layernorm = nn.LayerNorm(config.hidden_size)

    def _embed(self, x: mx.array) -> mx.array:
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
        embeddings += self.position_embedding
        return embeddings

    def __call__(self, x: mx.array) -> CLIPVisionOutput:
        x = self._embed(x)
        x = self.pre_layernorm(x)

        for l in self.layers:
            x = l(x, mask=None)

        # Extract <CLS> token embedding
        pooler_output = self.post_layernorm(x[:, 0, :])
        return CLIPVisionOutput(pooler_output=pooler_output, last_hidden_state=x)


class CLIPModel(nn.Module):
    def __init__(self, config: CLIPConfig):
        self.text_model = CLIPTextModel(config.text_config)
        self.vision_model = CLIPVisionModel(config.vision_config)

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
        )

        config = CLIPConfig(
            text_config=text_config,
            vision_config=vision_config,
            projection_dim=config["projection_dim"],
        )
        model = CLIPModel(config)
        model.load_weights(str(path / "weights.npz"))
        return model
