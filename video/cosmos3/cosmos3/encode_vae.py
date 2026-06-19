"""Standalone VAE encode: load HuggingFace VAE weights and encode image/video to latents.

Implements the Wan2.2 encoder architecture (inverse of decoder):
- Spatial patchification (pixel-space → patched input)
- WanResidualDownBlock: resnets + downsample + AvgDown3D residual shortcut
- Mid-block self-attention
- quant_conv → mean extraction (argmax mode)
- Per-channel normalization

Multi-frame encoding uses chunked processing with feat_cache to match HF's
causal temporal convolution behavior: frame 0 alone, then 4 frames at a time.
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .decode_vae import (
    _transpose_conv3d_weight,
    _transpose_conv2d_weight,
    _conv3d_forward,
    _rms_norm,
    _resnet_block,
    _attention_block,
)

# Match HF CACHE_T = 2: each causal conv caches the last 2 frames of its input
CACHE_T = 2


def _conv3d_forward_cached(
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    feat_cache: list,
    feat_idx: list,
    stride=(1, 1, 1),
    padding=(1, 1, 1),
) -> mx.array:
    """Conv3D with feat_cache for chunked temporal processing.

    Mirrors HF's WanCausalConv3d.forward(x, cache_x) + the cache bookkeeping
    done in WanEncoder3d.forward / WanResidualBlock.forward.

    Cache protocol (matching HF exactly):
    - Before the conv, save the last CACHE_T frames of the input as new cache
    - If previous cache exists and current input has <2 temporal frames,
      prepend the last frame of the previous cache
    - Use the previous cache as temporal context instead of zero-padding

    Args:
        x: [B, T, H, W, C] channels-last input
        weight: [O, kD, kH, kW, I] conv weight
        bias: [O] or None
        feat_cache: mutable list of cached activations
        feat_idx: mutable [int] index into feat_cache
        stride: (sD, sH, sW)
        padding: (pD, pH, pW)
    """
    idx = feat_idx[0]
    causal_pad_t = 2 * padding[0]

    # Cache bookkeeping: save last CACHE_T frames of input before conv
    cache_x = x[:, -CACHE_T:] if x.shape[1] >= CACHE_T else x
    if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        # Prepend last frame from previous chunk's cache
        cache_x = mx.concatenate([feat_cache[idx][:, -1:], cache_x], axis=1)

    # Build temporal context for the convolution
    if feat_cache[idx] is not None and causal_pad_t > 0:
        # Use cached frames instead of zero padding
        prev_cache = feat_cache[idx]
        # HF: x = torch.cat([cache_x, x], dim=2) where cache_x is the previous cache
        x = mx.concatenate([prev_cache, x], axis=1)
        # Reduce the zero padding needed
        remaining_pad = causal_pad_t - prev_cache.shape[1]
        if remaining_pad > 0:
            b, _, h, w, c = x.shape
            pad_zeros = mx.zeros((b, remaining_pad, h, w, c), dtype=x.dtype)
            x = mx.concatenate([pad_zeros, x], axis=1)
    elif causal_pad_t > 0:
        # First chunk: standard causal zero-padding
        b, _, h, w, c = x.shape
        pad_zeros = mx.zeros((b, causal_pad_t, h, w, c), dtype=x.dtype)
        x = mx.concatenate([pad_zeros, x], axis=1)

    # Update cache for next chunk
    feat_cache[idx] = cache_x
    feat_idx[0] += 1

    # Run the actual convolution (no additional causal padding - already applied)
    b, t_padded, h, w, c = x.shape
    o_ch, kd, kh, kw, i_ch = weight.shape
    sd, sh, sw = stride

    t_out = (t_padded - kd) // sd + 1

    outputs = []
    for ti in range(t_out):
        t_start = ti * sd
        accum = None
        for d in range(kd):
            frame = x[:, t_start + d]
            w_2d = weight[:, d, :, :, :]
            conv_out = mx.conv2d(frame, w_2d, stride=(sh, sw),
                                 padding=(padding[1], padding[2]))
            accum = conv_out if accum is None else accum + conv_out
        if bias is not None:
            accum = accum + bias
        outputs.append(accum)

    return mx.stack(outputs, axis=1)


def _resnet_block_cached(
    x: mx.array,
    weights: dict,
    prefix: str,
    feat_cache: list,
    feat_idx: list,
) -> mx.array:
    """Residual block with feat_cache for chunked processing.

    Mirrors HF's WanResidualBlock.forward with feat_cache.
    Cache slots: conv1 uses one slot, conv2 uses one slot.
    conv_shortcut (1x1) doesn't need temporal caching (kernel_size=1).
    """
    residual = x

    # First: norm -> silu -> conv1 (cached)
    x = _rms_norm(x, weights[f"{prefix}.norm1.gamma"])
    x = nn.silu(x)
    x = _conv3d_forward_cached(
        x, weights[f"{prefix}.conv1.weight"], weights.get(f"{prefix}.conv1.bias"),
        feat_cache, feat_idx,
    )
    mx.eval(x)

    # Second: norm -> silu -> conv2 (cached)
    x = _rms_norm(x, weights[f"{prefix}.norm2.gamma"])
    x = nn.silu(x)
    x = _conv3d_forward_cached(
        x, weights[f"{prefix}.conv2.weight"], weights.get(f"{prefix}.conv2.bias"),
        feat_cache, feat_idx,
    )
    mx.eval(x)

    # Skip connection (conv_shortcut for channel changes — 1x1, no temporal cache needed)
    skip_key = f"{prefix}.conv_shortcut.weight"
    if skip_key in weights:
        residual = _conv3d_forward(residual,
                                   weights[skip_key],
                                   weights.get(f"{prefix}.conv_shortcut.bias"),
                                   padding=(0, 0, 0))

    return x + residual


def _wan_resample_downsample2d(x: mx.array, weights: dict, prefix: str) -> mx.array:
    """WanResample downsample2d: learned Conv2d with stride 2 per frame.

    Args:
        x: [B, T, H, W, C] channels-last
        weights: dict with keys like "{prefix}.resample.1.weight", "{prefix}.resample.1.bias"
    """
    b, t, h, w, c = x.shape

    frames = []
    for ti in range(t):
        frame = x[:, ti]  # [B, H, W, C]
        # HF uses ZeroPad2d((0,1,0,1)) → asymmetric padding: right and bottom
        # In channels-last: pad W (right) and H (bottom) by 1
        frame = mx.pad(frame, [(0, 0), (0, 1), (0, 1), (0, 0)])
        conv_w = weights[f"{prefix}.resample.1.weight"]
        conv_b = weights.get(f"{prefix}.resample.1.bias")
        frame = mx.conv2d(frame, conv_w, stride=(2, 2), padding=(0, 0))
        if conv_b is not None:
            frame = frame + conv_b
        frames.append(frame)

    return mx.stack(frames, axis=1)


def _wan_resample_downsample3d_cached(
    x: mx.array,
    weights: dict,
    prefix: str,
    feat_cache: list,
    feat_idx: list,
) -> mx.array:
    """WanResample downsample3d with feat_cache: spatial downsample + temporal time_conv.

    Mirrors HF's WanResample.forward for downsample3d mode with feat_cache:
    - First run: spatial downsample only. Store the spatially-downsampled output
      in feat_cache for next chunk. Skip time_conv.
    - Subsequent runs: prepend cached last frame, apply time_conv (stride-2 temporal
      downsample), then spatial downsample.

    Args:
        x: [B, T, H, W, C] channels-last
    """
    # Spatial downsample first (always runs)
    x_spatial = _wan_resample_downsample2d(x, weights, prefix)

    idx = feat_idx[0]
    if feat_cache[idx] is None:
        # First chunk: just store, skip time_conv
        feat_cache[idx] = x_spatial
        feat_idx[0] += 1
        return x_spatial
    else:
        # Subsequent chunks: apply time_conv with cached context
        cache_x = x_spatial[:, -1:]  # save last frame for next chunk

        # Prepend last frame of previous cache
        prev_last = feat_cache[idx][:, -1:]
        x_with_context = mx.concatenate([prev_last, x_spatial], axis=1)

        # time_conv: WanCausalConv3d(dim, dim, (3,1,1), stride=(2,1,1), padding=(0,0,0))
        # kernel_size=(3,1,1), stride=(2,1,1), padding=(0,0,0)
        # With causal padding = 2*0 = 0 (no causal pad for this conv)
        # The cache provides the temporal context instead
        tc_weight = weights[f"{prefix}.time_conv.weight"]
        tc_bias = weights.get(f"{prefix}.time_conv.bias")

        x_out = _conv3d_forward(
            x_with_context, tc_weight, tc_bias,
            stride=(2, 1, 1), padding=(0, 0, 0), causal=False,
        )

        feat_cache[idx] = cache_x
        feat_idx[0] += 1
        return x_out


def _nearest_downsample_2x(x_2d: mx.array) -> mx.array:
    """Strided 2x spatial downsampling for a 2D tensor [B, H, W, C]."""
    return x_2d[:, ::2, ::2, :]


def _avg_down_3d(x: mx.array, in_channels: int, out_channels: int,
                 factor_t: int, factor_s: int) -> mx.array:
    """AvgDown3D: parameter-free channel-reshaping average-pool residual shortcut.

    Matches HF's AvgDown3D exactly. Reshapes input by interleaving spatial/temporal
    factors into channels, groups, and averages to produce the target channel count
    at downsampled resolution.

    Args:
        x: [B, T, H, W, C] channels-last
        in_channels: input channel count
        out_channels: output channel count
        factor_t: temporal downsampling factor (1 or 2)
        factor_s: spatial downsampling factor (1 or 2)

    Returns:
        [B, T//factor_t, H//factor_s, W//factor_s, out_channels]
    """
    factor = factor_t * factor_s * factor_s
    group_size = in_channels * factor // out_channels

    b, t, h, w, c = x.shape

    # Pad temporal if needed
    pad_t = (factor_t - t % factor_t) % factor_t
    if pad_t > 0:
        x = mx.pad(x, [(0, 0), (pad_t, 0), (0, 0), (0, 0), (0, 0)])
        t = t + pad_t

    t_out = t // factor_t
    h_out = h // factor_s
    w_out = w // factor_s

    # [B, T, H, W, C] -> [B, T//ft, ft, H//fs, fs, W//fs, fs, C]
    x = x.reshape(b, t_out, factor_t, h_out, factor_s, w_out, factor_s, c)
    # Move factors next to C: [B, T', H', W', C, ft, fs, fs]
    x = mx.transpose(x, (0, 1, 3, 5, 7, 2, 4, 6))
    # Collapse factors into channels: [B, T', H', W', C*factor]
    x = x.reshape(b, t_out, h_out, w_out, c * factor)
    # Group and average: [B, T', H', W', out_channels, group_size]
    x = x.reshape(b, t_out, h_out, w_out, out_channels, group_size)
    x = mx.mean(x, axis=-1)

    return x


def _patchify_input(x: mx.array, patch_size: int = 2) -> mx.array:
    """Patchify pixel-space input: pack spatial patches into channels.

    Input: [B, T, H, W, 3]
    Output: [B, T, H//p, W//p, 3*p*p]

    Matches HF's patchify() exactly:
    HF: [B,C,T,H,W] -> view [B,C,T,H//p,p,W//p,p] -> permute [B,C,p,p,T,H//p,W//p]
        -> view [B, C*p*p, T, H//p, W//p]

    In channels-last: [B,T,H,W,C] -> [B,T,H//p,p,W//p,p,C]
        -> permute to [B,T,H//p,W//p,C,p,p] -> [B,T,H//p,W//p,C*p*p]
    """
    b, t, h, w, c = x.shape
    p = patch_size
    # [B, T, H, W, C] -> [B, T, H//p, p, W//p, p, C]
    x = x.reshape(b, t, h // p, p, w // p, p, c)
    # From [B,T,H//p,p_h,W//p,p_w,C] -> [B,T,H//p,W//p,C,p_w,p_h]
    x = mx.transpose(x, (0, 1, 2, 4, 6, 5, 3))
    x = x.reshape(b, t, h // p, w // p, c * p * p)
    return x


def _count_encoder_cache_slots(weights: dict, config: dict) -> int:
    """Count the number of feat_cache slots needed for the encoder.

    Each WanCausalConv3d in the encoder gets one cache slot.
    Matches HF's sum(isinstance(m, WanCausalConv3d) for m in encoder.modules()).
    """
    count = 0
    is_residual = config.get("is_residual", False)
    dim_mult = config.get("dim_mult", [1, 2, 4, 4])
    num_res_blocks = config.get("num_res_blocks", 2)
    temporal_downsample = config.get("temperal_downsample", [False, True, True])

    # conv_in: 1 CausalConv3d
    count += 1

    # down_blocks
    base_dim = config.get("base_dim", 160)
    dims = [base_dim * u for u in [1] + dim_mult]

    for block_idx in range(len(dim_mult)):
        in_dim = dims[block_idx]
        out_dim = dims[block_idx + 1]
        down_flag = block_idx != len(dim_mult) - 1

        if is_residual:
            # WanResidualDownBlock: num_res_blocks resnets
            for _ in range(num_res_blocks):
                # Each WanResidualBlock: conv1 + conv2 = 2 CausalConv3d
                count += 2
                # conv_shortcut (1x1) exists when in_dim != out_dim but uses
                # non-cached _conv3d_forward (kernel_size=1, no temporal context
                # needed), so it does NOT consume a cache slot.
                in_dim = out_dim

            # downsampler
            if down_flag:
                t_down = temporal_downsample[block_idx] if block_idx < len(temporal_downsample) else False
                if t_down:
                    # downsample3d: time_conv is a CausalConv3d
                    count += 1
                # Note: the spatial Conv2d in the resampler is NOT a CausalConv3d
        else:
            # Non-residual path (shouldn't apply for Cosmos3)
            for _ in range(num_res_blocks):
                count += 2
                if in_dim != out_dim:
                    count += 1
                in_dim = out_dim
            if down_flag:
                t_down = temporal_downsample[block_idx] if block_idx < len(temporal_downsample) else False
                if t_down:
                    count += 1

    # mid_block: num_layers=1 means 2 resnets (initial + 1 per layer)
    # Each resnet: conv1 + conv2 = 2 CausalConv3d
    count += 2  # resnets[0]
    count += 2  # resnets[1]

    # conv_out: 1 CausalConv3d
    count += 1

    return count


def _encoder_forward(
    x: mx.array,
    weights: dict,
    config: dict,
    feat_cache: list | None = None,
    feat_idx: list | None = None,
) -> mx.array:
    """Run the encoder forward pass, optionally with feat_cache for chunked processing.

    Args:
        x: [B, T, H, W, C] already-patchified input
        weights: encoder weights dict (keys without 'encoder.' prefix)
        config: VAE config dict
        feat_cache: list of cached activations (None for single-pass mode)
        feat_idx: mutable [int] index into feat_cache (None for single-pass mode)
    """
    cached = feat_cache is not None

    # conv_in
    if cached:
        x = _conv3d_forward_cached(
            x, weights["conv_in.weight"], weights.get("conv_in.bias"),
            feat_cache, feat_idx,
        )
    else:
        x = _conv3d_forward(x, weights["conv_in.weight"], weights.get("conv_in.bias"))
    mx.eval(x)

    # Architecture config
    is_residual = config.get("is_residual", False)
    temporal_downsample = config.get("temperal_downsample", [False, True, True])
    dim_mult = config.get("dim_mult", [1, 2, 4, 4])
    base_dim = config.get("base_dim", 160)

    dims = [base_dim * u for u in [1] + dim_mult]
    num_down_blocks = len(dim_mult)

    for block_idx in range(num_down_blocks):
        prefix = f"down_blocks.{block_idx}"
        in_dim = dims[block_idx]
        out_dim = dims[block_idx + 1]

        # Save input for residual shortcut (WanResidualDownBlock)
        x_copy = x if is_residual else None

        # Count resnets
        resnet_ids = set()
        for k in weights:
            if k.startswith(f"{prefix}.resnets."):
                ri = int(k.split(".")[3])
                resnet_ids.add(ri)
        num_resnets = len(resnet_ids)

        for ri in range(num_resnets):
            if cached:
                x = _resnet_block_cached(x, weights, f"{prefix}.resnets.{ri}",
                                         feat_cache, feat_idx)
            else:
                x = _resnet_block(x, weights, f"{prefix}.resnets.{ri}")
            mx.eval(x)

        # Downsampling
        has_downsampler = any(k.startswith(f"{prefix}.downsampler.") for k in weights)
        down_flag = block_idx != len(dim_mult) - 1
        t_down = temporal_downsample[block_idx] if block_idx < len(temporal_downsample) and down_flag else False

        if has_downsampler:
            has_time_conv = f"{prefix}.downsampler.time_conv.weight" in weights
            if has_time_conv and t_down:
                if cached:
                    x = _wan_resample_downsample3d_cached(
                        x, weights, f"{prefix}.downsampler", feat_cache, feat_idx)
                else:
                    # Single-pass: skip time_conv (HF behavior when feat_cache is None)
                    x = _wan_resample_downsample2d(x, weights, f"{prefix}.downsampler")
            else:
                x = _wan_resample_downsample2d(x, weights, f"{prefix}.downsampler")
            mx.eval(x)

        # AvgDown3D residual shortcut (WanResidualDownBlock)
        if is_residual:
            factor_t = 2 if t_down else 1
            factor_s = 2 if down_flag else 1
            shortcut = _avg_down_3d(x_copy, in_dim, out_dim, factor_t, factor_s)
            x = x + shortcut
            mx.eval(x)

    # mid_block: resnet[0] -> attention[0] -> resnet[1]
    if cached:
        x = _resnet_block_cached(x, weights, "mid_block.resnets.0", feat_cache, feat_idx)
    else:
        x = _resnet_block(x, weights, "mid_block.resnets.0")
    mx.eval(x)

    attn_key = "mid_block.attentions.0.to_qkv.weight"
    if attn_key in weights:
        x = _attention_block(x, weights, "mid_block.attentions.0")
        mx.eval(x)

    if cached:
        x = _resnet_block_cached(x, weights, "mid_block.resnets.1", feat_cache, feat_idx)
    else:
        x = _resnet_block(x, weights, "mid_block.resnets.1")
    mx.eval(x)

    # norm_out + silu + conv_out
    x = _rms_norm(x, weights["norm_out.gamma"])
    x = nn.silu(x)
    if cached:
        x = _conv3d_forward_cached(
            x, weights["conv_out.weight"], weights.get("conv_out.bias"),
            feat_cache, feat_idx,
        )
    else:
        x = _conv3d_forward(x, weights["conv_out.weight"], weights.get("conv_out.bias"))
    mx.eval(x)

    return x


def _load_encoder_weights(vae_dir: str):
    """Load and prepare encoder weights and config from VAE directory.

    Returns:
        (weights, qc_weight, qc_bias, config) tuple
    """
    vae_path = Path(vae_dir)

    with open(vae_path / "config.json") as f:
        config = json.load(f)

    raw_weights = mx.load(str(vae_path / "diffusion_pytorch_model.safetensors"))

    weights = {}
    qc_weight = None
    qc_bias = None
    for k, v in raw_weights.items():
        if k == "quant_conv.weight":
            qc_weight = _transpose_conv3d_weight(v).astype(mx.bfloat16)
            continue
        if k == "quant_conv.bias":
            qc_bias = v.astype(mx.bfloat16)
            continue
        if not k.startswith("encoder."):
            continue
        name = k[len("encoder."):]

        if "conv" in name and name.endswith(".weight") and v.ndim == 5:
            v = _transpose_conv3d_weight(v)
        elif name.endswith(".weight") and v.ndim == 4:
            v = _transpose_conv2d_weight(v)

        weights[name] = v.astype(mx.bfloat16)

    return weights, qc_weight, qc_bias, config


def _prepare_input(video: np.ndarray | mx.array) -> mx.array:
    """Prepare video input: normalize to [-1, 1], ensure [1, T, H, W, 3] shape."""
    if isinstance(video, np.ndarray):
        if video.dtype == np.uint8:
            video = video.astype(np.float32) / 255.0
        x = mx.array(video)
    else:
        x = video

    if x.ndim == 3:
        x = mx.expand_dims(mx.expand_dims(x, 0), 0)  # [H,W,3] -> [1, 1, H, W, 3]
    elif x.ndim == 4:
        x = mx.expand_dims(x, 0)  # [T,H,W,3] -> [1, T, H, W, 3]
    x = x * 2.0 - 1.0  # [0,1] -> [-1,1]
    x = x.astype(mx.bfloat16)
    return x


def encode_video(
    video: np.ndarray | mx.array,
    vae_dir: str,
) -> mx.array:
    """Encode a video (or single image) to normalized VAE latents.

    Uses chunked encoding matching HF's _encode(): frame 0 processed alone,
    then 4 frames at a time, with feat_cache propagating causal temporal
    convolution state between chunks.

    For single-frame input, runs without chunking (no feat_cache overhead).

    Args:
        video: [T, H, W, 3] or [H, W, 3] uint8/float32 in [0, 1].
        vae_dir: path to vae/ directory with config.json and safetensors

    Returns:
        [1, T_lat, H//16, W//16, z_dim] normalized latents (channels-last)
    """
    weights, qc_weight, qc_bias, config = _load_encoder_weights(vae_dir)
    x = _prepare_input(video)

    num_frames = x.shape[1]

    # Patchify
    patch_size = config.get("patch_size", 2)
    if patch_size is not None and patch_size > 1:
        x = _patchify_input(x, patch_size)
    if num_frames == 1:
        # Single frame: no chunking needed, no feat_cache overhead
        out = _encoder_forward(x, weights, config)
    else:
        # Multi-frame: chunked encoding matching HF's _encode()
        # iter_ = 1 + (num_frame - 1) // 4
        # Chunk 0: frame 0 alone
        # Chunk i (i>0): frames 1+4*(i-1) : 1+4*i
        num_cache_slots = _count_encoder_cache_slots(weights, config)
        feat_cache = [None] * num_cache_slots
        feat_idx = [0]

        # Chunk 0: first frame
        feat_idx[0] = 0
        out = _encoder_forward(x[:, :1], weights, config, feat_cache, feat_idx)

        # Subsequent chunks: 4 frames at a time
        num_iter = 1 + (num_frames - 1 + 3) // 4  # ceil division: process all frames
        for i in range(1, num_iter):
            feat_idx[0] = 0
            start = 1 + 4 * (i - 1)
            end = min(1 + 4 * i, num_frames)
            chunk = x[:, start:end]
            chunk_out = _encoder_forward(chunk, weights, config, feat_cache, feat_idx)
            out = mx.concatenate([out, chunk_out], axis=1)
            mx.eval(out)

    # quant_conv
    if qc_weight is not None:
        out = _conv3d_forward(out, qc_weight, qc_bias,
                              stride=(1, 1, 1), padding=(0, 0, 0), causal=False)
        mx.eval(out)

    # Extract mean (argmax mode): first z_dim channels
    z_dim = config.get("z_dim", 48)
    mu = out[..., :z_dim]

    # Normalize: z_norm = (mu - mean) * inv_std
    latents_mean = config.get("latents_mean", [0.0] * z_dim)
    latents_std = config.get("latents_std", [1.0] * z_dim)
    mean = mx.array(latents_mean, dtype=mu.dtype)
    inv_std = 1.0 / mx.array(latents_std, dtype=mu.dtype)
    z_norm = (mu - mean) * inv_std

    return z_norm


# Backward-compatible alias
encode_image = encode_video
