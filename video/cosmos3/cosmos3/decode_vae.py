"""Standalone VAE decode: load HuggingFace VAE weights and decode latents to pixels.

Implements the full WanResidualUpBlock decoder architecture including:
- Mid-block self-attention (WanAttentionBlock)
- Residual upsampling blocks with DupUp3D shortcut
- Learned Conv2d after nearest-exact interpolation (WanResample)
- Temporal upsampling via CausalConv3d
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _transpose_conv3d_weight(w: mx.array) -> mx.array:
    """Transpose Conv3D weight from PyTorch [O,I,D,H,W] to MLX-friendly [O,D,H,W,I]."""
    return mx.transpose(w, (0, 2, 3, 4, 1))


def _transpose_conv2d_weight(w: mx.array) -> mx.array:
    """Transpose Conv2D weight from PyTorch [O,I,H,W] to MLX [O,H,W,I]."""
    return mx.transpose(w, (0, 2, 3, 1))


def _conv3d_forward(x: mx.array, weight: mx.array, bias: mx.array = None,
                    stride=(1,1,1), padding=(1,1,1), causal=True) -> mx.array:
    """Run Conv3D via per-frame 2D decomposition.

    Args:
        x: [B, T, H, W, C] channels-last
        weight: [O, kD, kH, kW, I] MLX layout
        bias: [O] optional
        stride: (sD, sH, sW)
        padding: (pD, pH, pW)
        causal: if True, use causal temporal padding
    """
    b, t, h, w, c = x.shape
    o_ch, kd, kh, kw, i_ch = weight.shape
    sd, sh, sw = stride

    # Causal temporal padding
    if causal:
        pad_t = 2 * padding[0]
        pad_zeros = mx.zeros((b, pad_t, h, w, c), dtype=x.dtype)
        x = mx.concatenate([pad_zeros, x], axis=1)
    else:
        # Symmetric padding
        pad_t = padding[0]
        if pad_t > 0:
            pad_zeros = mx.zeros((b, pad_t, h, w, c), dtype=x.dtype)
            x = mx.concatenate([pad_zeros, x, pad_zeros], axis=1)

    t_padded = x.shape[1]
    t_out = (t_padded - kd) // sd + 1

    outputs = []
    for ti in range(t_out):
        t_start = ti * sd
        accum = None
        for d in range(kd):
            frame = x[:, t_start + d]  # [B, H, W, C]
            w_2d = weight[:, d, :, :, :]  # [O, kH, kW, I]
            conv_out = mx.conv2d(frame, w_2d, stride=(sh, sw),
                                 padding=(padding[1], padding[2]))
            accum = conv_out if accum is None else accum + conv_out
        if bias is not None:
            accum = accum + bias
        outputs.append(accum)

    return mx.stack(outputs, axis=1)


def _rms_norm(x: mx.array, gamma: mx.array, eps: float = 1e-6) -> mx.array:
    """RMS norm with gamma. Gamma may have trailing singleton dims."""
    g = gamma.reshape(-1)
    rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
    return x / rms * g


def _resnet_block(x: mx.array, weights: dict, prefix: str) -> mx.array:
    """Run one residual block."""
    residual = x

    x = _rms_norm(x, weights[f"{prefix}.norm1.gamma"])
    x = nn.silu(x)
    x = _conv3d_forward(x,
                        weights[f"{prefix}.conv1.weight"],
                        weights.get(f"{prefix}.conv1.bias"))
    mx.eval(x)

    x = _rms_norm(x, weights[f"{prefix}.norm2.gamma"])
    x = nn.silu(x)
    x = _conv3d_forward(x,
                        weights[f"{prefix}.conv2.weight"],
                        weights.get(f"{prefix}.conv2.bias"))
    mx.eval(x)

    # Skip connection (conv_shortcut for channel changes)
    skip_key = f"{prefix}.conv_shortcut.weight"
    if skip_key in weights:
        residual = _conv3d_forward(residual,
                                   weights[skip_key],
                                   weights.get(f"{prefix}.conv_shortcut.bias"),
                                   padding=(0, 0, 0))

    return x + residual


def _attention_block(x: mx.array, weights: dict, prefix: str) -> mx.array:
    """Run self-attention block (WanAttentionBlock).

    Operates per-frame: reshapes [B, T, H, W, C] to [B*T, H, W, C],
    applies norm → qkv → attention → proj, then reshapes back.

    Args:
        x: [B, T, H, W, C] channels-last
        weights: dict with keys like "{prefix}.norm.gamma", "{prefix}.to_qkv.weight", etc.
    """
    identity = x
    b, t, h, w, c = x.shape

    # Reshape to per-frame: [B*T, H, W, C]
    x = x.reshape(b * t, h, w, c)

    # RMS norm
    x = _rms_norm(x, weights[f"{prefix}.norm.gamma"])

    # QKV projection via 1x1 conv: [B*T, H, W, C] -> [B*T, H, W, 3*C]
    # HF uses Conv2d(C, 3*C, 1) which is a 1x1 conv
    qkv_w = weights[f"{prefix}.to_qkv.weight"]  # [3C, 1, 1, C] in MLX layout
    qkv_b = weights.get(f"{prefix}.to_qkv.bias")
    qkv = mx.conv2d(x, qkv_w, padding=(0, 0))
    if qkv_b is not None:
        qkv = qkv + qkv_b

    # Reshape for attention: [B*T, H*W, 3*C] -> split to q, k, v each [B*T, H*W, C]
    qkv = qkv.reshape(b * t, h * w, 3 * c)
    q, k, v = mx.split(qkv, 3, axis=-1)

    # Single-head scaled dot-product attention
    # [B*T, 1, H*W, C] for SDPA
    scale = c ** -0.5
    q = mx.expand_dims(q, 1)
    k = mx.expand_dims(k, 1)
    v = mx.expand_dims(v, 1)
    attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    attn = attn.squeeze(1)  # [B*T, H*W, C]

    # Reshape back to spatial: [B*T, H, W, C]
    attn = attn.reshape(b * t, h, w, c)

    # Output projection via 1x1 conv
    proj_w = weights[f"{prefix}.proj.weight"]
    proj_b = weights.get(f"{prefix}.proj.bias")
    out = mx.conv2d(attn, proj_w, padding=(0, 0))
    if proj_b is not None:
        out = out + proj_b

    # Reshape back to [B, T, H, W, C]
    out = out.reshape(b, t, h, w, c)

    return out + identity


def _nearest_upsample_2x(x_2d: mx.array) -> mx.array:
    """Nearest-neighbor 2x upsampling for a 2D tensor [B, H, W, C]."""
    # Repeat along H and W
    b, h, w, c = x_2d.shape
    x_2d = mx.repeat(x_2d, 2, axis=1)  # [B, 2H, W, C]
    x_2d = mx.repeat(x_2d, 2, axis=2)  # [B, 2H, 2W, C]
    return x_2d


def _wan_resample_upsample2d(x: mx.array, weights: dict, prefix: str) -> mx.array:
    """WanResample upsample2d: nearest-exact interpolation + learned Conv2d per frame.

    Args:
        x: [B, T, H, W, C] channels-last
        weights: dict with keys like "{prefix}.resample.1.weight", "{prefix}.resample.1.bias"
    """
    b, t, h, w, c = x.shape

    # Process per-frame
    frames = []
    for ti in range(t):
        frame = x[:, ti]  # [B, H, W, C]

        # 2x nearest upsample
        frame = _nearest_upsample_2x(frame)  # [B, 2H, 2W, C]

        # Learned Conv2d(C, out_dim, 3, padding=1)
        conv_w = weights[f"{prefix}.resample.1.weight"]  # [O, kH, kW, I]
        conv_b = weights.get(f"{prefix}.resample.1.bias")
        frame = mx.conv2d(frame, conv_w, padding=(1, 1))
        if conv_b is not None:
            frame = frame + conv_b

        frames.append(frame)

    return mx.stack(frames, axis=1)


def _wan_resample_upsample3d(x: mx.array, weights: dict, prefix: str) -> mx.array:
    """WanResample upsample3d: temporal conv doubling + spatial upsample + conv.

    Temporal: CausalConv3d(C, 2C, (3,1,1)) → reshape to interleave → doubles T
    Spatial: nearest 2x + Conv2d

    Args:
        x: [B, T, H, W, C] channels-last
        weights: dict with keys for time_conv and resample
    """
    b, t, h, w, c = x.shape

    # Temporal upsampling via CausalConv3d(C, 2C, (3,1,1), padding=(1,0,0))
    time_conv_w = weights[f"{prefix}.time_conv.weight"]  # [2C, kD, 1, 1, C]
    time_conv_b = weights.get(f"{prefix}.time_conv.bias")
    x_t = _conv3d_forward(x, time_conv_w, time_conv_b,
                          stride=(1, 1, 1), padding=(1, 0, 0), causal=True)
    mx.eval(x_t)
    # x_t: [B, T, H, W, 2C]
    # Reshape to interleave frames: split channels in half, interleave along time
    x_t = x_t.reshape(b, t, h, w, 2, c)
    # Interleave: [B, T, H, W, 2, C] -> [B, 2T, H, W, C]
    x_t = mx.transpose(x_t, (0, 1, 4, 2, 3, 5))  # [B, T, 2, H, W, C]
    x_t = x_t.reshape(b, t * 2, h, w, c)

    # Spatial upsampling: nearest 2x + learned Conv2d per frame
    x = _wan_resample_upsample2d(x_t, weights, prefix)
    return x


def _dup_up_3d_residual(x: mx.array, in_c: int, out_c: int,
                        factor_t: int, factor_s: int) -> mx.array:
    """DupUp3D residual shortcut: channel-repeat + reshape for skip connection.

    HF DupUp3D: repeat_interleave on channels, then reshape to interleave spatially.

    Args:
        x: [B, T, H, W, C] channels-last (PyTorch: [B, C, T, H, W])
        in_c: input channels
        out_c: output channels
        factor_t: temporal upsample factor (1 or 2)
        factor_s: spatial upsample factor (2)
    """
    b, t, h, w, c = x.shape
    factor = factor_t * factor_s * factor_s
    repeats = out_c * factor // in_c

    # repeat_interleave on channel dim
    x = mx.repeat(x, repeats, axis=-1)  # [B, T, H, W, C*repeats]

    # Reshape to expose upsample factors
    # PyTorch: [B, out_c, factor_t, factor_s, factor_s, T, H, W]
    # MLX channels-last: [B, T, H, W, out_c, factor_t, factor_s, factor_s]
    x = x.reshape(b, t, h, w, out_c, factor_t, factor_s, factor_s)

    # Permute to interleave spatial/temporal factors
    # Target: [B, T*factor_t, H*factor_s, W*factor_s, out_c]
    # Intermediate: [B, T, factor_t, H, factor_s, W, factor_s, out_c]
    x = mx.transpose(x, (0, 1, 5, 2, 6, 3, 7, 4))
    x = x.reshape(b, t * factor_t, h * factor_s, w * factor_s, out_c)

    return x


def decode_latents(
    latents: mx.array,
    vae_dir: str,
    latents_mean: list[float] = None,
    latents_std: list[float] = None,
) -> mx.array:
    """Decode latents to video frames using HuggingFace VAE weights directly.

    Implements the full WanResidualUpBlock decoder architecture.

    Args:
        latents: [B, T, H, W, z_dim] denoised latents (channels-last)
        vae_dir: path to vae/ directory with config.json and safetensors
        latents_mean: per-channel mean for denormalization
        latents_std: per-channel std for denormalization

    Returns:
        [B, T_out, H_out, W_out, 3] decoded video frames in [0, 1]
    """
    vae_path = Path(vae_dir)

    # Load config
    with open(vae_path / "config.json") as f:
        config = json.load(f)

    # Load weights
    raw_weights = mx.load(str(vae_path / "diffusion_pytorch_model.safetensors"))

    # Extract decoder weights and post_quant_conv, transpose Conv3D and Conv2D
    weights = {}
    pqc_weight = None
    pqc_bias = None
    for k, v in raw_weights.items():
        if k == "post_quant_conv.weight":
            # [O, I, 1, 1, 1] -> [O, 1, 1, 1, I] for Conv3D, but since kernel=1
            # it's effectively a linear transform on channels
            pqc_weight = _transpose_conv3d_weight(v).astype(mx.bfloat16)
            continue
        if k == "post_quant_conv.bias":
            pqc_bias = v.astype(mx.bfloat16)
            continue
        if not k.startswith("decoder."):
            continue
        name = k[len("decoder."):]

        # Transpose Conv3D weights: [O,I,D,H,W] -> [O,D,H,W,I]
        if "conv" in name and name.endswith(".weight") and v.ndim == 5:
            v = _transpose_conv3d_weight(v)
        # Transpose Conv2D weights: [O,I,H,W] -> [O,H,W,I]
        elif name.endswith(".weight") and v.ndim == 4:
            v = _transpose_conv2d_weight(v)

        weights[name] = v.astype(mx.bfloat16)

    # Denormalize latents
    if latents_mean is None:
        latents_mean = config.get("latents_mean", [0.0] * latents.shape[-1])
    if latents_std is None:
        latents_std = config.get("latents_std", [1.0] * latents.shape[-1])

    mean = mx.array(latents_mean, dtype=latents.dtype)
    std = mx.array(latents_std, dtype=latents.dtype)
    z = latents * std + mean

    # Track input temporal dimension for T=1 vs T>1 behavior differences
    input_t = z.shape[1]

    # post_quant_conv: 1x1x1 Conv3D that transforms latent channels before decoder
    if pqc_weight is not None:
        z = _conv3d_forward(z, pqc_weight, pqc_bias,
                            stride=(1, 1, 1), padding=(0, 0, 0), causal=False)
        mx.eval(z)

    # conv_in
    x = _conv3d_forward(z, weights["conv_in.weight"], weights.get("conv_in.bias"))
    mx.eval(x)

    # mid_block: resnet[0] -> attention[0] -> resnet[1]
    x = _resnet_block(x, weights, "mid_block.resnets.0")
    mx.eval(x)

    # Mid-block attention (if weights exist)
    attn_key = "mid_block.attentions.0.to_qkv.weight"
    if attn_key in weights:
        x = _attention_block(x, weights, "mid_block.attentions.0")
        mx.eval(x)

    x = _resnet_block(x, weights, "mid_block.resnets.1")
    mx.eval(x)

    # up_blocks (WanResidualUpBlock)
    temporal_upsample = list(reversed(config.get("temperal_downsample", [False, True, True])))
    is_residual = config.get("is_residual", False)

    # Count up_blocks from weights
    up_block_ids = set()
    for k in weights:
        if k.startswith("up_blocks."):
            idx = int(k.split(".")[1])
            up_block_ids.add(idx)
    num_up_blocks = len(up_block_ids)

    for block_idx in range(num_up_blocks):
        prefix = f"up_blocks.{block_idx}"

        # Save input for residual shortcut
        x_copy = x

        # Count resnets in this block
        resnet_ids = set()
        for k in weights:
            if k.startswith(f"{prefix}.resnets."):
                ri = int(k.split(".")[3])
                resnet_ids.add(ri)
        num_resnets = len(resnet_ids)

        # Run resnets
        for ri in range(num_resnets):
            x = _resnet_block(x, weights, f"{prefix}.resnets.{ri}")
            mx.eval(x)

        # Upsampling (WanResample) if upsampler weights exist
        has_upsampler = any(k.startswith(f"{prefix}.upsampler.") for k in weights)
        t_up = temporal_upsample[block_idx] if block_idx < len(temporal_upsample) else False

        if has_upsampler:
            has_time_conv = f"{prefix}.upsampler.time_conv.weight" in weights
            if has_time_conv and t_up and input_t > 1:
                # Multi-frame: full 3D upsample (spatial + temporal)
                x = _wan_resample_upsample3d(x, weights, f"{prefix}.upsampler")
            else:
                # Single-frame or no temporal: spatial-only 2D upsample.
                # HF's cached path skips time_conv when feat_cache[idx] is None
                # (first call); for T=1 single-pass this is the correct behavior.
                x = _wan_resample_upsample2d(x, weights, f"{prefix}.upsampler")
            mx.eval(x)

        # DupUp3D residual shortcut + add
        # DupUp3D has no learned weights — it's a pure channel-reshape operation.
        # For T=1: HF uses first_chunk=True which applies factor_t=2 then trims
        # the first frame. For T>1: full temporal expansion, no trim.
        if is_residual and has_upsampler:
            in_c = x_copy.shape[-1]
            out_c = x.shape[-1]
            factor_t = 2 if t_up else 1
            factor_s = 2
            shortcut = _dup_up_3d_residual(x_copy, in_c, out_c, factor_t, factor_s)
            # first_chunk trim for single-frame decode
            if factor_t > 1 and input_t == 1:
                shortcut = shortcut[:, factor_t - 1:, :, :, :]
            x = x + shortcut
            mx.eval(x)

    # norm_out + silu + conv_out
    x = _rms_norm(x, weights["norm_out.gamma"])
    x = nn.silu(x)
    x = _conv3d_forward(x, weights["conv_out.weight"], weights.get("conv_out.bias"))
    mx.eval(x)

    # Unpatchify if needed (patch_size=2)
    # HF packs channels as [C, p1, p2] and interleaves H with p2, W with p1.
    # In channels-last: reshape to [B, T, H, W, C, p1, p2], then permute so
    # p2 (dim6) interleaves with H and p1 (dim5) interleaves with W.
    patch_size = config.get("patch_size", 2)
    if patch_size > 1:
        b, t, h, w, c_patch = x.shape
        c = c_patch // (patch_size * patch_size)
        x = x.reshape(b, t, h, w, c, patch_size, patch_size)
        x = mx.transpose(x, (0, 1, 2, 6, 3, 5, 4))  # [B, T, H, p2, W, p1, C]
        x = x.reshape(b, t, h * patch_size, w * patch_size, c)

    # Clamp to [0, 1]
    x = (mx.clip(x, -1.0, 1.0) + 1.0) / 2.0

    return x
