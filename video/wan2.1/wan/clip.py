# Copyright © 2026 Apple Inc.

"""
CLIP ViT-H/14 vision encoder for Wan2.1 I2V.

Ported from the OpenCLIP XLM-RoBERTa CLIP model. Only the visual encoder is
used — text encoder, post_norm, and projection head are discarded.

Architecture: ViT-H/14 (image_size=224, patch_size=14, dim=1280, heads=16,
layers=32, mlp_ratio=4, pool=token, activation=gelu, pre_norm=True).

At inference the first 31 of 32 transformer blocks are used
(use_31_block=True). Block 31's weights are loaded but never evaluated.
"""

import re

import mlx.core as mx
import mlx.nn as nn


class CLIPAttentionBlock(nn.Module):
    """Pre-norm transformer block with self-attention and MLP."""

    def __init__(self, dim: int = 1280, num_heads: int = 16, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = _CLIPSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _CLIPMLP(dim, int(dim * mlp_ratio))

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.self_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _CLIPSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        B, L, _ = x.shape
        n, d = self.num_heads, self.head_dim
        q = self.q_proj(x).reshape(B, L, n, d).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, n, d).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, n, d).transpose(0, 2, 1, 3)
        x = mx.fast.scaled_dot_product_attention(q, k, v, scale=d**-0.5)
        x = x.transpose(0, 2, 1, 3).reshape(B, L, n * d)
        return self.out_proj(x)


class _CLIPMLP(nn.Module):
    def __init__(self, dim: int, mid_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class CLIPVisionEncoder(nn.Module):
    """ViT-H/14 vision encoder returning patch + CLS token features."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        dim: int = 1280,
        num_heads: int = 16,
        num_layers: int = 32,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2  # 256
        self.dim = dim
        self.num_layers = num_layers

        self.patch_embedding = nn.Conv2d(
            3, dim, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.cls_embedding = mx.zeros((1, 1, dim))
        self.position_embedding = mx.zeros((1, self.num_patches + 1, dim))
        self.pre_norm = nn.LayerNorm(dim)

        # All 32 blocks loaded; only first 31 used in forward (see __call__).
        # Block 31 weights are loaded but never evaluated.
        for i in range(num_layers):
            setattr(self, f"block_{i}", CLIPAttentionBlock(dim, num_heads, mlp_ratio))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [B, 224, 224, 3] preprocessed image (channels-last).

        Returns:
            [B, 257, 1280] CLS + patch token features.
        """
        B = x.shape[0]

        # Patch embed: [B, H, W, 3] -> [B, H', W', dim] -> [B, num_patches, dim]
        x = self.patch_embedding(x)
        x = x.reshape(B, -1, self.dim)

        cls = mx.broadcast_to(self.cls_embedding, (B, 1, self.dim))
        x = mx.concatenate([cls, x], axis=1)
        x = x + self.position_embedding
        x = self.pre_norm(x)

        # Only first 31 of 32 blocks (matching reference use_31_block=True)
        for i in range(self.num_layers - 1):
            block = getattr(self, f"block_{i}")
            x = block(x)

        return x

    @staticmethod
    def sanitize(weights):
        """Remap CLIP .pth checkpoint keys to MLX model format.

        Handles both standard OpenCLIP naming and Wan2.1 HF naming.
        Extracts only visual.* keys. Splits fused QKV into q/k/v.
        Discards post_norm, head, and all non-visual keys.
        """
        remapped = {}
        for key, value in weights.items():
            if not key.startswith("visual."):
                continue

            # Skip post_norm, head
            if "post_norm" in key or "ln_post" in key or key == "visual.head":
                continue

            # patch_embedding
            if key in ("visual.conv1.weight", "visual.patch_embedding.weight"):
                if value.ndim == 4:
                    value = mx.transpose(value, (0, 2, 3, 1))
                remapped["patch_embedding.weight"] = value
                continue

            # CLS embedding
            if key in ("visual.class_embedding", "visual.cls_embedding"):
                remapped["cls_embedding"] = value.reshape(1, 1, -1)
                continue

            # Position embedding
            if key in ("visual.positional_embedding", "visual.pos_embedding"):
                if value.ndim == 2:
                    value = value.reshape(1, value.shape[0], value.shape[1])
                remapped["position_embedding"] = value
                continue

            # Pre-norm (both "visual.ln_pre.*" and "visual.pre_norm.*")
            if key.startswith("visual.ln_pre.") or key.startswith("visual.pre_norm."):
                param = key.split(".")[-1]
                remapped[f"pre_norm.{param}"] = value
                continue

            # Transformer blocks — handle both naming conventions:
            #   OpenCLIP:  visual.transformer.resblocks.N.*
            #   Wan2.1 HF: visual.transformer.N.*
            m = re.match(r"visual\.transformer\.(?:resblocks\.)?(\d+)\.(.*)", key)
            if m:
                block_idx = m.group(1)
                rest = m.group(2)

                # Fused QKV: "attn.in_proj_*" or "attn.to_qkv.*"
                if rest in ("attn.in_proj_weight", "attn.to_qkv.weight"):
                    dim = value.shape[0] // 3
                    q, k, v = value[:dim], value[dim : 2 * dim], value[2 * dim :]
                    remapped[f"block_{block_idx}.self_attn.q_proj.weight"] = q
                    remapped[f"block_{block_idx}.self_attn.k_proj.weight"] = k
                    remapped[f"block_{block_idx}.self_attn.v_proj.weight"] = v
                    continue

                if rest in ("attn.in_proj_bias", "attn.to_qkv.bias"):
                    dim = value.shape[0] // 3
                    q, k, v = value[:dim], value[dim : 2 * dim], value[2 * dim :]
                    remapped[f"block_{block_idx}.self_attn.q_proj.bias"] = q
                    remapped[f"block_{block_idx}.self_attn.k_proj.bias"] = k
                    remapped[f"block_{block_idx}.self_attn.v_proj.bias"] = v
                    continue

                # Out projection: "attn.out_proj.*" or "attn.proj.*"
                for prefix in ("attn.out_proj.", "attn.proj."):
                    if rest.startswith(prefix):
                        param = rest.split(".")[-1]
                        remapped[f"block_{block_idx}.self_attn.out_proj.{param}"] = (
                            value
                        )
                        break
                else:
                    # Norms: "ln_1.*" or "norm1.*", "ln_2.*" or "norm2.*"
                    for old, new in [
                        ("ln_1.", "norm1."),
                        ("norm1.", "norm1."),
                        ("ln_2.", "norm2."),
                        ("norm2.", "norm2."),
                    ]:
                        if rest.startswith(old):
                            param = rest.split(".")[-1]
                            remapped[f"block_{block_idx}.{new}{param}"] = value
                            break
                    else:
                        # MLP: "mlp.c_fc.*" or "mlp.0.*", "mlp.c_proj.*" or "mlp.2.*"
                        for old, new in [
                            ("mlp.c_fc.", "mlp.fc1."),
                            ("mlp.0.", "mlp.fc1."),
                            ("mlp.c_proj.", "mlp.fc2."),
                            ("mlp.2.", "mlp.fc2."),
                        ]:
                            if rest.startswith(old):
                                param = rest.split(".")[-1]
                                remapped[f"block_{block_idx}.{new}{param}"] = value
                                break

        return remapped


def preprocess_clip_image(image_path: str) -> mx.array:
    """Load and preprocess an image for CLIP ViT-H/14.

    The reference CLIP visual() receives images in [-1, 1], then does
    mul_(0.5).add_(0.5) to get [0, 1], then applies normalize. We replicate
    this: load -> resize 224x224 -> [0,1] -> normalize.

    Returns:
        [1, 224, 224, 3] float32 array (channels-last for MLX).
    """
    from PIL import Image

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224), Image.BICUBIC)

    # Convert to float32 [0, 1]
    import numpy as np

    arr = np.array(img).astype(np.float32) / 255.0

    # Normalize per-channel
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)

    return mx.array(arr[np.newaxis])  # [1, 224, 224, 3]
