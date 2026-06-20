"""Standalone audio decode: load HuggingFace sound tokenizer weights and decode audio latents.

Implements the Oobleck decoder with weight-normalized convolutions and SnakeBeta activations.
Loads weights directly from safetensors — no nn.Module tree needed.
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _weight_norm_conv1d(x: mx.array, weight_v: mx.array, weight_g: mx.array,
                        bias: mx.array = None, stride: int = 1,
                        padding: int = 0, dilation: int = 1) -> mx.array:
    """Apply weight-normalized Conv1d.

    Weight normalization: w = g * (v / ||v||)
    PyTorch weight_v: [out_ch, in_ch, kernel] -> MLX: [out_ch, kernel, in_ch]

    Args:
        x: [B, C, T] channels-first
        weight_v: [out_ch, kernel, in_ch] MLX layout (already transposed)
        weight_g: [out_ch, 1, 1] magnitude
        bias: [out_ch] optional
    """
    # Compute normalized weight: g * v / ||v||
    # Norm over (in_ch, kernel) dims — axes 1,2 in MLX layout
    v_norm = mx.sqrt(mx.sum(weight_v * weight_v, axis=(1, 2), keepdims=True) + 1e-12)
    weight = weight_g * weight_v / v_norm

    # MLX conv1d expects [B, T, C] channels-last
    x_cl = mx.transpose(x, (0, 2, 1))  # [B, T, C]
    out = mx.conv1d(x_cl, weight, stride=stride, padding=padding, dilation=dilation)
    if bias is not None:
        out = out + bias
    return mx.transpose(out, (0, 2, 1))  # back to [B, C, T]


def _weight_norm_conv_transpose1d(x: mx.array, weight_v: mx.array, weight_g: mx.array,
                                  bias: mx.array = None, stride: int = 1,
                                  padding: int = 0) -> mx.array:
    """Apply weight-normalized ConvTranspose1d.

    PyTorch ConvTranspose1d weight: [in_ch, out_ch, kernel]
    MLX ConvTranspose1d weight: [out_ch, kernel, in_ch]

    Args:
        x: [B, C, T] channels-first
        weight_v: [out_ch, kernel, in_ch] MLX layout (already transposed)
        weight_g: [in_ch, 1, 1] magnitude (norm over out_ch*kernel per input channel)
        bias: [out_ch] optional
    """
    # Weight norm: g * v / ||v|| — for ConvTranspose1d, PyTorch weight_g is [in_ch, 1, 1]
    # norming over (out_ch, kernel). In MLX layout [out_ch, kernel, in_ch], that's axes (0, 1).
    v_norm = mx.sqrt(mx.sum(weight_v * weight_v, axis=(0, 1), keepdims=True) + 1e-12)
    g = weight_g.reshape(1, 1, -1)  # [in_ch, 1, 1] -> [1, 1, in_ch]
    weight = g * weight_v / v_norm

    # MLX conv_transpose1d: input [B, T, C_in], weight [C_out, K, C_in]
    x_cl = mx.transpose(x, (0, 2, 1))  # [B, T, C]
    out = mx.conv_transpose1d(x_cl, weight, stride=stride, padding=padding)
    if bias is not None:
        out = out + bias
    return mx.transpose(out, (0, 2, 1))  # [B, C, T]


def _snake_beta(x: mx.array, alpha: mx.array, beta: mx.array,
                log_scale: bool = True) -> mx.array:
    """SnakeBeta activation: x + (1/b) * sin²(a*x).

    Args:
        x: [B, C, T] channels-first
        alpha: [1, C, 1]
        beta: [1, C, 1]
        log_scale: if True, alpha and beta are in log space
    """
    if log_scale:
        a = mx.exp(alpha)
        b = mx.exp(beta)
    else:
        a = alpha
        b = beta
    return x + (1.0 / (b + 1e-9)) * mx.power(mx.sin(a * x), 2)


def decode_audio(
    audio_latents: mx.array,
    sound_tokenizer_dir: str,
) -> mx.array:
    """Decode audio latents to waveform using HF sound tokenizer weights.

    Args:
        audio_latents: [B, latent_dim, T_latent] or [latent_dim, T_latent]
            Audio latents from the diffusion model (channels-first)
        sound_tokenizer_dir: path to sound_tokenizer/ directory

    Returns:
        [B, 2, T_audio] stereo waveform in [-1, 1]
    """
    if audio_latents.ndim == 2:
        audio_latents = mx.expand_dims(audio_latents, 0)

    snd_path = Path(sound_tokenizer_dir)

    with open(snd_path / "config.json") as f:
        config = json.load(f)

    raw_weights = mx.load(str(snd_path / "diffusion_pytorch_model.safetensors"))

    # Extract and transpose decoder weights
    weights = {}
    for k, v in raw_weights.items():
        if not k.startswith("decoder."):
            continue
        name = k[len("decoder."):]

        # Transpose Conv1d weight_v: PyTorch [O, I, K] -> MLX [O, K, I]
        if name.endswith(".weight_v") and v.ndim == 3:
            # For ConvTranspose1d (conv_t1), PyTorch shape is [I, O, K]
            # For Conv1d, PyTorch shape is [O, I, K]
            if "conv_t" in name:
                # ConvTranspose1d: [I, O, K] -> MLX [O, K, I]
                v = mx.transpose(v, (1, 2, 0))
            else:
                # Conv1d: [O, I, K] -> MLX [O, K, I]
                v = mx.transpose(v, (0, 2, 1))

        weights[name] = v.astype(mx.float32)

    log_scale = config.get("snake_logscale", True)
    strides = config.get("dec_strides", [2, 4, 5, 6, 8])

    x = audio_latents.astype(mx.float32)

    # conv1: Conv1d(64, 5120, 7, padding=3) with weight norm
    x = _weight_norm_conv1d(x, weights["conv1.weight_v"], weights["conv1.weight_g"],
                            weights.get("conv1.bias"), padding=3)
    mx.eval(x)

    # Decoder blocks (5 blocks, strides reversed = [8, 6, 5, 4, 2])
    for block_idx in range(5):
        prefix = f"block.{block_idx}"
        stride = strides[-(block_idx + 1)]  # reverse order

        # Snake activation before upsample
        x = _snake_beta(x,
                        weights[f"{prefix}.snake1.alpha"],
                        weights[f"{prefix}.snake1.beta"],
                        log_scale)

        # ConvTranspose1d upsample
        pad = stride // 2
        x = _weight_norm_conv_transpose1d(
            x,
            weights[f"{prefix}.conv_t1.weight_v"],
            weights[f"{prefix}.conv_t1.weight_g"],
            weights.get(f"{prefix}.conv_t1.bias"),
            stride=stride, padding=pad,
        )
        mx.eval(x)

        # 3 residual units
        for unit_idx in range(1, 4):
            unit_prefix = f"{prefix}.res_unit{unit_idx}"
            residual = x

            # Snake1 + Conv1 (dilated)
            x = _snake_beta(x,
                            weights[f"{unit_prefix}.snake1.alpha"],
                            weights[f"{unit_prefix}.snake1.beta"],
                            log_scale)

            # Dilation pattern: 1, 3, 9
            dilation = 3 ** (unit_idx - 1)
            pad_d = dilation * 3  # kernel=7, (7-1)//2 * dilation
            x = _weight_norm_conv1d(
                x,
                weights[f"{unit_prefix}.conv1.weight_v"],
                weights[f"{unit_prefix}.conv1.weight_g"],
                weights.get(f"{unit_prefix}.conv1.bias"),
                padding=pad_d, dilation=dilation,
            )

            # Snake2 + Conv2 (1x1)
            x = _snake_beta(x,
                            weights[f"{unit_prefix}.snake2.alpha"],
                            weights[f"{unit_prefix}.snake2.beta"],
                            log_scale)
            x = _weight_norm_conv1d(
                x,
                weights[f"{unit_prefix}.conv2.weight_v"],
                weights[f"{unit_prefix}.conv2.weight_g"],
                weights.get(f"{unit_prefix}.conv2.bias"),
            )

            # Trim to match residual
            if x.shape[-1] > residual.shape[-1]:
                x = x[..., :residual.shape[-1]]
            elif x.shape[-1] < residual.shape[-1]:
                residual = residual[..., :x.shape[-1]]

            x = x + residual

        mx.eval(x)

    # Final: snake1 + conv2 (output conv)
    x = _snake_beta(x, weights["snake1.alpha"], weights["snake1.beta"], log_scale)
    # conv2 is Conv1d(320, 2, 7, padding=3) — no bias key in the weights
    x = _weight_norm_conv1d(x, weights["conv2.weight_v"], weights["conv2.weight_g"],
                            weights.get("conv2.bias"), padding=3)
    mx.eval(x)

    x = mx.clip(x, -1.0, 1.0)
    return x
