import glob
import inspect
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from llama import LlamaModel, TextConfig

from clip import ClipVisionModel, VisionConfig


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
        self.linear_1 = nn.Linear(config.in_features, config.out_features)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(config.out_features, config.out_features)

    def forward(self, x: mx.array) -> mx.array:
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        return x


class LlavaModel(nn.Module):
    def __init__(self, config: LlaVAConfig):
        self.vision_tower = ClipVisionModel(
            config=VisionConfig.from_dict(config.vision_config)
        )
        self.language_model = LlamaModel(args=TextConfig.from_dict(config.text_config))
        self.multi_modal_projector = LlavaMultiModalProjector(
            config=config.projection_config
        )

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
    ):
        # TODO: add the forward pass

        if pixel_values is not None and input_ids.shape[1] != 1:
            image_outputs = self.vision_tower(pixel_values)

            # TODO: this is not the correct output layer, but it's a placeholder
            selected_image_feature = image_outputs.pooler_output

            image_features = self.multi_modal_projector(selected_image_feature)

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids, attention_mask, labels
    ):
        # TODO: https://github.com/huggingface/transformers/blob/4f09d0fd888dbf2660313f9715992822acfb99ce/src/transformers/models/llava/modeling_llava.py#L279

        special_image_token_mask = input_ids == self.config.special_tokens.image

        num_image_tokens = special_image_token_mask.sum()

        pass

    @staticmethod
    def from_pretrained(path: str):
        path = Path(path)

        with open(path / "config.json", "r") as f:
            model_config = json.load(f)

        model_config = LlaVAConfig.from_dict(model_config)
        model = LlavaModel(model_config)
        weight_files = glob.glob(str(path / "*.safetensors"))
        if not weight_files:
            logging.error(f"No safetensors found in {path}")
            raise FileNotFoundError(f"No safetensors found in {path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        weights = ClipVisionModel.sanitize(weights)
        model.load_weights(list(weights.items()))
        return model
