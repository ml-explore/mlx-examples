import glob
import inspect
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from language import LanguageModel, TextConfig
from utils import get_model_path
from vision import VisionConfig, VisionModel


@dataclass
class LlaVAConfig:
    text_config: TextConfig
    vision_config: VisionConfig
    ignore_index: int = -100
    image_token_index: int = 32000
    vision_feature_select_strategy: str = "default"
    vision_feature_layer: int = -2
    vocab_size: int = 32000

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config: LlaVAConfig):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size, config.text_config.hidden_size, bias=True
        )

        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=True
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        return x


class LlavaModel(nn.Module):
    def __init__(self, config: LlaVAConfig):
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.vision_feature_layer = config.vision_feature_layer
        self.vision_feature_select_strategy = config.vision_feature_select_strategy

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
    ):
        if pixel_values is None:
            return self.language_model(input_ids)

        # Get the input embeddings from the language model
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        # get the ouptut hidden states from the vision model
        _, _, hidden_states = self.vision_tower(
            pixel_values.transpose(0, 2, 3, 1), output_hidden_states=True
        )
        # select the hidden states from the desired layer
        selected_image_feature = hidden_states[self.vision_feature_layer]

        if self.vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif self.vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(
                f"Unexpected select feature strategy: {self.vision_feature_select_strategy}"
            )
        # pass image features through the multi-modal projector
        image_features = self.multi_modal_projector(selected_image_feature)
        # insert special image tokens in the input_ids
        final_inputs_embeds = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids
        )

        return final_inputs_embeds

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids
    ):
        image_token_index = self.config.image_token_index
        num_images, num_image_patches, embed_dim = image_features.shape

        # Positions of <image> tokens in `input_ids` and assume batch size is 1
        image_positions = [
            idx
            for idx, token_id in enumerate(input_ids[0])
            if token_id == image_token_index
        ]

        if len(image_positions) != num_images:
            raise ValueError(
                f"The input provided to the model is incorrect. The number of image tokens is {len(image_positions)}, but the number of images given to the model is {num_images}."
            )

        text_segments = []
        start_idx = 0

        for position in image_positions:
            text_segments.append(inputs_embeds[:, start_idx:position])
            start_idx = position + 1

        if start_idx < inputs_embeds.shape[1]:
            text_segments.append(inputs_embeds[:, start_idx:])

        # Reshape image feature from (num_images, num_image_patches, embed_dim) to (num_images*num_image_patches, embed_dim)
        image_embeddings = image_features.reshape(-1, image_features.shape[-1])

        final_embeddings = []
        for i, text_segment in enumerate(text_segments):
            final_embeddings.append(text_segment[0])
            if i < len(image_positions):
                # Add a slice of image embeddings corresponding to the current position.
                # This effectively replaces one <image> token with its associated num_image_patches embeddings.
                final_embeddings.append(image_embeddings[i : i + num_image_patches + 1])

        # This creates a final embeding in shape (1, num_image_patches*num_images + sequence_len, embed_dim) representing the merged sequence of text and image embeddings.
        final_embeddings = mx.concatenate(final_embeddings, axis=0).reshape(
            1, -1, embed_dim
        )

        return final_embeddings

    def __call__(self, input_ids: mx.array, pixel_values: mx.array, cache=None):
        input_embddings = self.get_input_embeddings(input_ids, pixel_values)
        logits, cache = self.language_model(
            input_ids, cache=cache, inputs_embeds=input_embddings
        )
        return logits, cache

    @staticmethod
    def from_pretrained(path: str):
        path = get_model_path(path)
        path = Path(path)

        with open(path / "config.json", "r") as f:
            model_config = json.load(f)

        model_config = LlaVAConfig.from_dict(model_config)

        if isinstance(model_config.vision_config, dict):
            model_config.vision_config = VisionConfig.from_dict(
                model_config.vision_config
            )

        if isinstance(model_config.text_config, dict):
            model_config.text_config = TextConfig.from_dict(model_config.text_config)

        model = LlavaModel(model_config)
        weight_files = glob.glob(str(path / "*.safetensors"))
        if not weight_files:
            logging.error(f"No safetensors found in {path}")
            raise FileNotFoundError(f"No safetensors found in {path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        if hasattr(VisionModel, "sanitize"):
            weights = VisionModel.sanitize(weights)

        if hasattr(VisionModel, "sanitize"):
            weights = LanguageModel.sanitize(weights)

        model.load_weights(list(weights.items()))
        return model
