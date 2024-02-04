from dataclasses import dataclass
from typing import Literal, Optional

MLX_WEIGHTS_PATH = "weights/mlx"
JAX_WEIGHTS_PATH = "weights/jax"


@dataclass
class MlpMixerConfig:
    img_size: int
    patch_size: int
    num_blocks: int
    hidden_dim: int
    tokens_mlp_dim: int
    channels_mlp_dim: int
    num_classes: Optional[int] = None


def get_mlp_mixer_config(variant: Literal = ["B", "L"], num_classes: int = 1000):
    # Hyperparameters taken from the paper: https://arxiv.org/pdf/2105.01601.pdf
    if variant == "B":
        return MlpMixerConfig(
            img_size=224,
            patch_size=16,
            num_blocks=12,
            hidden_dim=768,
            tokens_mlp_dim=384,
            channels_mlp_dim=3072,
            num_classes=num_classes,
        )
    else:
        return MlpMixerConfig(
            img_size=224,
            patch_size=16,
            num_blocks=24,
            hidden_dim=1024,
            tokens_mlp_dim=512,
            channels_mlp_dim=4096,
            num_classes=num_classes,
        )


MODELS = {
    "imagenet1k-MixerB-16": {
        "config": get_mlp_mixer_config("B", 1000),
        "weights": "gs://mixer_models/imagenet1k/Mixer-B_16.npz",
    },
    "imagenet1k-MixerL-16": {
        "config": get_mlp_mixer_config("L", 1000),
        "weights": "gs://mixer_models/imagenet1k/Mixer-L_16.npz",
    },
    "imagenet21k-MixerB-16": {
        "config": get_mlp_mixer_config("B", 21843),
        "weights": "gs://mixer_models/imagenet21k/Mixer-B_16.npz",
    },
    "imagenet21k-MixerL-16": {
        "config": get_mlp_mixer_config("L", 21843),
        "weights": "gs://mixer_models/imagenet21k/Mixer-L_16.npz",
    },
}
