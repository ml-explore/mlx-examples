from clip import CLIPVisionModel
from llama import LlamaModel
from pathlib import Path
import json
import mlx.nn as nn
import mlx.core as mx
from typing import Any, Optional, Dict, Union


from dataclasses import dataclass


@dataclass
class VisionConfig:
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_channels: int
    image_size: int
    patch_size: int


@dataclass
class LLMConfig:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    rope_theta: float = 10000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None


@dataclass
class ProjectionConfig:
    in_features: int
    out_features: int


@dataclass
class LlaVAConfig:
    llm_config: Any
    vision_config: VisionConfig
    projection_config: ProjectionConfig


class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config: Any):
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
        self.vision_tower = CLIPVisionModel(config=config.vision_config)
        self.language_model = LlamaModel(args=config.llm_config)
        self.multi_modal_projector = LlavaMultiModalProjector(
            config=config.projection_config)

    def __call__(self,
                 input_ids: Optional[mx.array] = None,
                 pixel_values: Optional[mx.array] = None):
        # TODO: add the forward pass

        if pixel_values is not None and input_ids.shape[1] != 1:
            image_outputs = self.vision_tower(pixel_values)

            # TODO: this is not the correct output layer, but it's a placeholder
            selected_image_feature = image_outputs.pooler_output

            image_features = self.multi_modal_projector(
                selected_image_feature)

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        # TODO: https://github.com/huggingface/transformers/blob/4f09d0fd888dbf2660313f9715992822acfb99ce/src/transformers/models/llava/modeling_llava.py#L279

        special_image_token_mask = input_ids == self.config.special_tokens.image

        num_image_tokens = special_image_token_mask.sum()

        pass

    @staticmethod
    def from_pretrained(path: str):
        path = Path(path)

        with open(path / "config.json", "r") as f:
            config = json.load(f)

        model = LlavaModel(config)
        model.load_weights(str(path / "weights.npz"))

        return model
