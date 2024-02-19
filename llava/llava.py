from clip import CLIPVisionModel
from llama import Llama
from pathlib import Path
import json
import mlx.nn as nn
import mlx.core as mx
from typing import Any, Optional


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
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int
    rope_theta: float
    rope_traditional: bool = True


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
        self.language_model = Llama(args=config.llm_config)
        self.multi_modal_projector = LlavaMultiModalProjector(
            config=config.projection_config)

    def __call__(self,
                 input_ids: Optional[mx.array] = None,
                 pixel_values: Optional[mx.array] = None):
        # TODO: add the forward pass
        pass

    @staticmethod
    def from_pretrained(path: str):
        path = Path(path)

        with open(path / "config.json", "r") as f:
            config = json.load(f)

        model = LlavaModel(config)
        model.load_weights(str(path / "weights.npz"))

        return model
