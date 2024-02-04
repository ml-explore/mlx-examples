import mlx.core as mx
import mlx.nn as nn
from config import MLX_WEIGHTS_PATH, MODELS, MlpMixerConfig


class MixerGELU(nn.Module):
    """GELU approximation from JAX."""

    def __call__(self, x: mx.array) -> mx.array:
        sqrt_2_over_pi = mx.sqrt(mx.array(2 / mx.pi)).astype(x.dtype)
        cdf = 0.5 * (1.0 + mx.tanh(sqrt_2_over_pi * (x + 0.044715 * (x**3))))
        return x * cdf


class MlpBlock(nn.Sequential):
    """Mixer block layer."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super(MlpBlock, self).__init__(
            nn.Linear(input_dim, hidden_dim),
            MixerGELU(),
            nn.Linear(hidden_dim, input_dim),
        )


class MixerBlock(nn.Module):
    """Mixer block layer."""

    def __init__(
        self,
        num_patches: int,
        hidden_dim: int,
        tokens_mlp_dim: int,
        channels_mlp_dim: int,
    ):
        self.token_mixing = MlpBlock(input_dim=num_patches, hidden_dim=tokens_mlp_dim)
        self.channel_mixing = MlpBlock(
            input_dim=hidden_dim, hidden_dim=channels_mlp_dim
        )
        self.ln0 = nn.LayerNorm(dims=hidden_dim, eps=1e-6)
        self.ln1 = nn.LayerNorm(dims=hidden_dim, eps=1e-6)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.ln0(x)
        # Mix tokens
        y = mx.swapaxes(y, 1, 2)
        y = self.token_mixing(y)
        y = mx.swapaxes(y, 1, 2)
        x += y
        # Mix patches
        y = self.ln1(x)
        y = self.channel_mixing(y)
        y += x
        return y


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, hidden_dim: int):
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)  # shape: (b, h / p, w / p, hidden_dim)
        x = mx.flatten(x, 1, 2)  # shape: (b, num_patches, hidden_dim)
        return x


class MlpMixer(nn.Module):
    """Mixer architecture."""

    def __init__(self, config: MlpMixerConfig):
        assert config.img_size % config.patch_size == 0
        num_patches = int(config.img_size / config.patch_size) ** 2

        self.patch_embedding = PatchEmbed(
            patch_size=config.patch_size, hidden_dim=config.hidden_dim
        )
        self.blocks = [
            MixerBlock(
                num_patches=num_patches,
                hidden_dim=config.hidden_dim,
                tokens_mlp_dim=config.tokens_mlp_dim,
                channels_mlp_dim=config.channels_mlp_dim,
            )
            for _ in range(config.num_blocks)
        ]
        self.pre_head_ln = nn.LayerNorm(dims=config.hidden_dim, eps=1e-6)
        if config.num_classes:
            self.head = nn.Linear(
                input_dims=config.hidden_dim, output_dims=config.num_classes
            )

    def __call__(self, x: mx.array) -> mx.array:
        # Compute patch projections
        x = self.patch_embedding(x)
        # Push through mixer blocks
        for b in self.blocks:
            x = b(x)
        # Classify
        x = self.pre_head_ln(x)
        if "head" in self:
            # Average pool over patches
            x = mx.mean(x, axis=1)
            # Logits
            x = self.head(x)
        return x


def load(model_name: str, mlx_weights_path: str = MLX_WEIGHTS_PATH) -> MlpMixer:
    assert model_name in MODELS

    config = MODELS[model_name]["config"]
    model = MlpMixer(config)
    model.load_weights(f"{mlx_weights_path}/{model_name}.npz")

    return model
