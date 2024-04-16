import glob
import inspect
import json
import re
from dataclasses import dataclass
from functools import partial, reduce
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from language import LanguageModel, TextConfig
from PIL import Image
from transformers import AutoConfig
from transformers.image_processing_utils import get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)
from vision import VisionConfig, VisionModel


@dataclass
class LlaVAConfig:
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    auto_map: dict
    hidden_size: int
    mm_hidden_size: int
    mm_hidden_size: int
    mm_vision_tower: str
    mm_projector_type: str = "mlp2x_gelu"
    ignore_index: int = -100
    image_token_index: int = -200
    vocab_size: int = 151936

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class SigLipImageProcessor:
    def __init__(
        self,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        size=(384, 384),
        crop_size: Dict[str, int] = None,
        resample=PILImageResampling.BICUBIC,
        rescale_factor=1 / 255,
        data_format=ChannelDimension.FIRST,
    ):
        crop_size = (
            crop_size if crop_size is not None else {"height": 384, "width": 384}
        )
        crop_size = get_size_dict(
            crop_size, default_to_square=True, param_name="crop_size"
        )

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size

    def preprocess(self, images):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            assert isinstance(images, list)

        transforms = [
            convert_to_rgb,
            to_numpy_array,
            partial(
                resize,
                size=self.size,
                resample=self.resample,
                data_format=self.data_format,
            ),
            partial(rescale, scale=self.rescale_factor, data_format=self.data_format),
            partial(
                normalize,
                mean=self.image_mean,
                std=self.image_std,
                data_format=self.data_format,
            ),
            partial(
                to_channel_dimension_format,
                channel_dim=self.data_format,
                input_channel_dim=self.data_format,
            ),
        ]

        images = reduce(lambda x, f: [*map(f, x)], transforms, images)

        return images


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


class SigLipVisionTower(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.vision_tower = VisionModel(config)

    def __call__(
        self, x: mx.array, output_hidden_states: Optional[bool] = None
    ) -> mx.array:
        return self.vision_tower(x, output_hidden_states)


class NanoLlavaModel(nn.Module):
    def __init__(self, config: LlaVAConfig):
        self.model_type = config.model_type
        self.config = config

        self.vision_tower = SigLipVisionTower(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.mm_projector = LlavaMultiModalProjector(config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
    ):
        if pixel_values is None:
            return self.language_model(input_ids)

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        *_, hidden_state = self.vision_tower(
            pixel_values.transpose(0, 2, 3, 1), output_hidden_states=True
        )

        image_features = hidden_state[-1].astype(pixel_values.dtype)
        assert image_features.shape[-2] == 729

        image_features = self.mm_projector(image_features)

        final_inputs_embeds = self._prepare_inputs_for_multimodal(
            image_features, inputs_embeds, input_ids
        )
        return final_inputs_embeds

    def _prepare_inputs_for_multimodal(self, image_features, inputs_embeds, input_ids):
        image_token_index = self.config.image_token_index
        num_images, num_image_patches, embed_dim = image_features.shape

        # Positions of <image> tokens in input_ids, assuming batch size is 1
        image_positions = np.where(input_ids[0] == image_token_index)[0].tolist()

        if len(image_positions) != num_images:
            raise ValueError(
                f"The number of image tokens ({len(image_positions)}) does not "
                f" match the number of image inputs ({num_images})."
            )

        text_segments = []
        start_idx = 0

        for position in image_positions:
            text_segments.append(inputs_embeds[:, start_idx:position])
            start_idx = position + 1

        image_embeddings = mx.split(image_features, image_features.shape[0])
        final_embeddings = [v for p in zip(text_segments, image_embeddings) for v in p]
        final_embeddings += [inputs_embeds[:, start_idx:]]

        # Create a final embedding of shape
        # (1, num_image_patches*num_images + sequence_len, embed_dim)
        return mx.concatenate(final_embeddings, axis=1)

    def __call__(self, input_ids: mx.array, pixel_values: mx.array, cache=None):
        input_embeddings = self.get_input_embeddings(input_ids, pixel_values)
        logits, cache = self.language_model(
            inputs=input_ids, cache=cache, inputs_embeds=input_embeddings
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
            config = json.load(f)

        siglip_config = AutoConfig.from_pretrained(config["mm_vision_tower"])
        text_config = AutoConfig.from_pretrained(config["language_model"])
        siglip_config = siglip_config.to_dict()
        text_config = text_config.to_dict()
        config["vision_config"] = siglip_config["vision_config"]
        config["text_config"] = text_config

        model_config = LlaVAConfig.from_dict(config)
        model_config.vision_config = VisionConfig.from_dict(config["vision_config"])
        model_config.text_config = TextConfig.from_dict(config["text_config"])

        model = NanoLlavaModel(model_config)
        weight_files = glob.glob(str(path / "*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors found in {path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        weights = {
            (
                f"{k.split('.', 1)[1]}"
                if re.match(r"^model\.vision_tower", k)
                else (
                    f"mm_projector.linear_1.{k.split('.')[-1]}"
                    if re.match(r"^model\.mm_projector\.0", k)
                    else (
                        f"mm_projector.linear_2.{k.split('.')[-1]}"
                        if re.match(r"^model\.mm_projector\.2", k)
                        else (
                            f"language_model.model.{k}"
                            if re.match(r"^lm_head", k)
                            else (
                                f"language_model.{k}"
                                if re.match(r"^model\.(embed_tokens|norm|layers)", k)
                                else k
                            )
                        )
                    )
                )
            ): v
            for k, v in weights.items()
        }

        weights = {
            (
                f"vision_tower.vision_tower.vision_model.head.attention.in_proj.bias"
                if re.match(
                    r"^vision_tower\.vision_tower\.vision_model\.head\.attention\.in_proj_bias",
                    k,
                )
                else (
                    f"vision_tower.vision_tower.vision_model.head.attention.in_proj.weight"
                    if re.match(
                        r"^vision_tower\.vision_tower\.vision_model\.head\.attention\.in_proj_weight",
                        k,
                    )
                    else k
                )
            ): v
            for k, v in weights.items()
        }

        weights = VisionModel(model_config.vision_config).sanitize(weights=weights)
        weights = LanguageModel(model_config.text_config).sanitize(weights=weights)
        model.load_weights(list(weights.items()))
        return model
