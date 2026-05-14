# Copyright © 2024 Apple Inc.

import glob
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from language import LanguageModel, TextConfig
from vision import VisionConfig, VisionModel


@dataclass
class LlaVAConfig:
    text_config: TextConfig
    vision_config: VisionConfig
    ignore_index: int = -100
    image_token_index: int = 32000
    image_grid_pinpoints: Optional[list] = None
    image_seq_length: int = 576
    vision_feature_select_strategy: str = "default"
    vision_feature_layer: int = -2
    use_image_newline_parameter: bool = False
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
        if config.use_image_newline_parameter:
            self.image_newline = mx.zeros(config.text_config.hidden_size)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        image_sizes: Optional[mx.array] = None,
    ):
        # Get the input embeddings from the language model
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        if pixel_values is None:
            return inputs_embeds

        if pixel_values.ndim == 5:
            image_features = self.get_image_features(pixel_values, image_sizes)
        else:
            # Get the output hidden states from the vision model
            *_, hidden_states = self.vision_tower(
                pixel_values.transpose(0, 2, 3, 1), output_hidden_states=True
            )

            # Select the hidden states from the desired layer
            selected_image_feature = hidden_states[self.vision_feature_layer]

            if self.vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
            elif self.vision_feature_select_strategy == "full":
                selected_image_feature = selected_image_feature
            else:
                raise ValueError(
                    "Unexpected feature selection strategy: "
                    f"{self.vision_feature_select_strategy}"
                )

            # Pass image features through the multi-modal projector
            image_features = self.multi_modal_projector(selected_image_feature)

        # Insert special image tokens in the input_ids
        final_inputs_embeds = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids
        )
        return final_inputs_embeds

    def get_image_features(self, pixel_values: mx.array, image_sizes: mx.array):
        if image_sizes is None:
            raise ValueError("image_sizes must be provided for LLaVA-NeXT images.")

        if hasattr(image_sizes, "tolist"):
            image_sizes = image_sizes.tolist()
        image_sizes = np.array(image_sizes)
        image_num_patches = [
            image_size_to_num_patches(
                image_size=image_size,
                grid_pinpoints=self.config.image_grid_pinpoints,
                patch_size=self.config.vision_config.image_size,
            )
            for image_size in image_sizes
        ]

        pixel_values = [
            pixel_value[:num_patches]
            for pixel_value, num_patches in zip(pixel_values, image_num_patches)
        ]
        pixel_values = mx.concatenate(pixel_values, axis=0)

        *_, hidden_states = self.vision_tower(
            pixel_values.transpose(0, 2, 3, 1), output_hidden_states=True
        )
        selected_image_feature = hidden_states[self.vision_feature_layer]

        if self.vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif self.vision_feature_select_strategy != "full":
            raise ValueError(
                "Unexpected feature selection strategy: "
                f"{self.vision_feature_select_strategy}"
            )

        image_features = self.multi_modal_projector(selected_image_feature)
        split_image_features = []
        start = 0
        for num_patches in image_num_patches:
            split_image_features.append(image_features[start : start + num_patches])
            start += num_patches
        image_features = split_image_features
        image_features = self.pack_image_features(image_features, image_sizes)
        return mx.concatenate(image_features, axis=0)[None]

    def pack_image_features(self, image_features, image_sizes):
        new_image_features = []
        image_newline = getattr(self, "image_newline", None)

        for image_feature, image_size in zip(image_features, image_sizes):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = (
                    self.config.vision_config.image_size
                    // self.config.vision_config.patch_size
                )
                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_size,
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                image_feature = image_feature.reshape(
                    num_patch_height, num_patch_width, height, width, -1
                )
                image_feature = image_feature.transpose(4, 0, 2, 1, 3)
                image_feature = image_feature.reshape(
                    image_feature.shape[0],
                    num_patch_height * height,
                    num_patch_width * width,
                )
                image_feature = unpad_image(image_feature, image_size)

                if image_newline is not None:
                    newline = mx.broadcast_to(
                        image_newline[:, None, None],
                        (*image_feature.shape[:-1], 1),
                    )
                    image_feature = mx.concatenate(
                        (image_feature, newline.astype(image_feature.dtype)),
                        axis=-1,
                    )

                image_feature = image_feature.reshape(
                    image_feature.shape[0],
                    image_feature.shape[1] * image_feature.shape[2],
                ).T
                image_feature = mx.concatenate(
                    (base_image_feature, image_feature), axis=0
                )
            else:
                image_feature = image_feature[0]
                if image_newline is not None:
                    image_feature = mx.concatenate(
                        (
                            image_feature,
                            image_newline[None].astype(image_feature.dtype),
                        ),
                        axis=0,
                    )
            new_image_features.append(image_feature)

        return new_image_features

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids
    ):
        image_token_index = self.config.image_token_index
        if image_features.ndim == 3:
            image_features = image_features[0]
        num_image_patches = image_features.shape[0]

        # Positions of <image> tokens in input_ids, assuming batch size is 1
        image_positions = mx.array(
            np.where(input_ids[0] == image_token_index)[0], mx.uint32
        )

        if len(image_positions) != num_image_patches:
            raise ValueError(
                f"The number of image tokens ({len(image_positions)}) does not "
                f" match the number of image patches ({num_image_patches})."
            )

        inputs_embeds[0, image_positions] = image_features
        return inputs_embeds

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        image_sizes: Optional[mx.array] = None,
        cache=None,
    ):
        input_embddings = self.get_input_embeddings(
            input_ids, pixel_values, image_sizes=image_sizes
        )
        logits, cache = self.language_model(
            input_ids, cache=cache, inputs_embeds=input_embddings
        )
        return logits, cache

    @staticmethod
    def from_pretrained(path_or_hf_repo: str):
        path = Path(path_or_hf_repo)
        if not path.exists():
            path = Path(
                snapshot_download(
                    repo_id=path_or_hf_repo,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.py",
                        "tokenizer.model",
                        "*.tiktoken",
                    ],
                )
            )

        with open(path / "config.json", "r") as f:
            model_config = json.load(f)

        model_config = LlaVAConfig.from_dict(model_config)

        model_config.vision_config = VisionConfig.from_dict(model_config.vision_config)
        model_config.text_config = TextConfig.from_dict(model_config.text_config)

        model = LlavaModel(model_config)
        weight_files = glob.glob(str(path / "*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors found in {path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        weights = VisionModel.sanitize(weights)
        weights = LanguageModel.sanitize(weights)

        model.load_weights(list(weights.items()))
        return model


def select_best_resolution(original_size: tuple, possible_resolutions: list) -> tuple:
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width = int(original_width * scale)
        downscaled_height = int(original_height * scale)
        effective_resolution = min(
            downscaled_width * downscaled_height, original_width * original_height
        )
        wasted_resolution = height * width - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution
            and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    return best_fit


def image_size_to_num_patches(image_size, grid_pinpoints, patch_size: int):
    if grid_pinpoints is None:
        return 1

    height, width = select_best_resolution(image_size, grid_pinpoints)
    num_patches = 0
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            num_patches += 1
    return num_patches + 1


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    height, width = select_best_resolution(image_size, grid_pinpoints)
    return height // patch_size, width // patch_size


def unpad_image(tensor, original_size):
    original_height, original_width = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(round(original_height * scale_factor, 7))
        padding = (current_height - new_height) // 2
        return tensor[:, padding : current_height - padding, :]

    scale_factor = current_height / original_height
    new_width = int(round(original_width * scale_factor, 7))
    padding = (current_width - new_width) // 2
    return tensor[:, :, padding : current_width - padding]
