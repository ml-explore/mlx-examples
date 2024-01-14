# Copyright © 2023 Apple Inc.

from dataclasses import dataclass
from functools import reduce
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from config import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from mlx.core import linalg as LA
from mlx.nn.losses import cross_entropy


def quick_gelu(x: mx.array) -> mx.array:
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """
    return x * mx.sigmoid(1.702 * x)


def clip_loss(logits_per_text: mx.array, logits_per_image: mx.array) -> mx.array:
    [N, _] = logits_per_text.shape
    [M, _] = logits_per_image.shape
    caption_loss = cross_entropy(logits_per_text, mx.arange(N), reduction="mean")
    image_loss = cross_entropy(logits_per_image, mx.arange(M), reduction="mean")
    return (caption_loss + image_loss) / 2.0


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
        self.attention.query_proj.bias = mx.zeros(hidden_dim)
        self.attention.key_proj.bias = mx.zeros(hidden_dim)
        self.attention.value_proj.bias = mx.zeros(hidden_dim)
        self.attention.out_proj.bias = mx.zeros(hidden_dim)


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
        # Extract some shapes
        _, N = x.shape
        # Compute the embeddings
        embeddings = self.token_embedding(x)
        embeddings += self.position_embedding[:N]
        return embeddings

    def __call__(self, x: mx.array) -> CLIPTextOutput:
        # Extract some shapes
        B, N = x.shape
        eot_tokens = mx.argmax(x, axis=-1)
        # Look up embeddings
        x = self._embed(x)
        # Compute the causal mask
        mask = nn.MultiHeadAttention.create_additive_causal_mask(N, x.dtype)
        # Push through the transformer
        x = reduce(lambda x, l: l(x, mask), self.layers, x)
        # Apply the final layernorm
        last_hidden_state = self.final_layer_norm(x)
        pooler_output = last_hidden_state[mx.arange(B), eot_tokens]

        return CLIPTextOutput(
            pooler_output=pooler_output, last_hidden_state=last_hidden_state
        )


class CLIPVisionModel(nn.Module):
    """Implements the vision encoder transformer from CLIP."""

    def __init__(self, config: CLIPVisionConfig):
        super().__init__()

        self.class_embedding = mx.random.normal((config.hidden_size,))
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        num_patches = (config.image_size // config.patch_size) ** 2
        num_positions = num_patches + 1
        self.position_embedding = mx.random.normal((num_positions, config.hidden_size))
        self.pre_layernorm = nn.LayerNorm(config.hidden_size)
        self.layers = [
            CLIPEncoderLayer(
                config.hidden_size, config.intermediate_size, config.num_attention_heads
            )
            for _ in range(config.num_hidden_layers)
        ]
        self.post_layernorm = nn.LayerNorm(config.hidden_size)

    def _embed(self, x: mx.array) -> mx.array:
        [batch_size, _, _, _] = x.shape
        # Patchify using conv; [batch_size, sqrt(num_patches), sqrt(num_patches), embed_dim]
        patch_embeddings = self.patch_embedding(x)
        # [batch_size, num_patches, embed_dim]
        patch_embeddings = mx.flatten(patch_embeddings, start_axis=1, end_axis=2)
        [_, _, embed_dim] = patch_embeddings.shape
        # Append <CLS> embeddings
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
        # Look up patch embeddings
        x = self._embed(x)
        # Prenorm
        x = self.pre_layernorm(x)
        # Push through transformer
        x = reduce(lambda x, l: l(x, mask=None), self.layers, x)
        # Pool <CLS> token
        pooler_output = self.post_layernorm(x[:, 0, :])
        return CLIPVisionOutput(pooler_output=pooler_output, last_hidden_state=x)


class CLIPModel(nn.Module):
    def __init__(self, config: CLIPConfig):
        if not isinstance(config.text_config, CLIPTextConfig):
            raise ValueError(
                f"config.text_config is expected to be of type CLIPTextConfig but is of type {type(config.text_config)}."
            )
        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise ValueError(
                f"config.vision_config is expected to be of type CLIPVisionConfig but is of type {type(config.vision_config)}."
            )

        self.text_model = CLIPTextModel(config.text_config)
        self.vision_model = CLIPVisionModel(config.vision_config)

        text_embed_dim = config.text_config.hidden_size
        vision_embed_dim = config.vision_config.hidden_size
        projection_dim = config.projection_dim

        self.visual_projection = nn.Linear(vision_embed_dim, projection_dim, bias=False)
        self.text_projection = nn.Linear(text_embed_dim, projection_dim, bias=False)
        self.logit_scale = mx.array([])

    def get_text_features(self, x: mx.array) -> mx.array:
        return self.text_projection(self.text_model(x).pooler_output)

    def get_image_features(self, x: mx.array) -> mx.array:
        return self.visual_projection(self.vision_model(x).pooler_output)

    def __call__(
        self, input_ids: mx.array, pixel_values: mx.array, return_loss=False
    ) -> Any:
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

        loss = None

        if input_ids is not None and pixel_values is not None:
            logit_scale = mx.exp(self.logit_scale)
            logits_per_text = (text_embeds @ image_embeds.T) * logit_scale
            logits_per_image = logits_per_text.T

            if return_loss:
                loss = clip_loss(logits_per_text, logits_per_image)

        return CLIPModelOutput(
            loss=loss,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            vision_model_output=vision_model_output,
            text_model_output=text_model_output,
        )
