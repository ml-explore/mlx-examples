"""
DINOv2-large image tokenizer with per-layer AdaLN modulation conditioned on
the camera embedding. Ported from SF3D's `sf3d/models/tokenizers/dinov2.py`
(a *vendored, modified* copy of `transformers.models.dinov2`, not the stock
HF module - do not try to reuse an off-the-shelf MLX DINOv2) plus
`sf3d/models/tokenizers/image.py`'s wrapper.

Config (facebook/dinov2-large, confirmed via its cached config.json):
  hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
  patch_size=14, mlp_ratio=4, layer_norm_eps=1e-6, layerscale_value=1.0,
  use_swiglu_ffn=false (plain GELU MLP). Trained at image_size=518 (37x37=
  1369 patches + 1 CLS = 1370 position embeddings), but SF3D feeds 512x512
  (512//14=36 -> 1296 patches + CLS = 1297 tokens) so position embeddings
  get bicubic-interpolated 37x37 -> 36x36 every forward pass (SF3D always
  uses a fixed 512 cond_image_size, so this interpolation is deterministic
  and could be precomputed, but ported 1:1 here for correctness first).
"""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .bicubic import bicubic_resize_nhwc

HIDDEN_SIZE = 1024
NUM_LAYERS = 24
NUM_HEADS = 16
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
PATCH_SIZE = 14
MLP_RATIO = 4
LAYER_NORM_EPS = 1e-6
MODULATION_COND_DIM = 768

IMAGE_MEAN = mx.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
IMAGE_STD = mx.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
# Force-materialize now, on whichever thread imports this module (module-
# level constants are normally imported on the main thread). Left lazy, the
# graph node for their construction is recorded against the importing
# thread's stream; combining them with an array built on a different thread
# (e.g. giu11.py's worker thread) and evaluating there then fails with
# "There is no Stream(gpu, 0) in current thread."
mx.eval(IMAGE_MEAN, IMAGE_STD)


class Modulation(nn.Module):
    """sf3d.models.transformers.attention.Modulation with single_layer=True
    (linear1 is nn.Identity() - the checkpoint has no linear1 weights)."""

    def __init__(self, embedding_dim: int, condition_dim: int):
        super().__init__()
        self.linear2 = nn.Linear(condition_dim, embedding_dim * 2)

    def __call__(self, x: mx.array, condition: mx.array) -> mx.array:
        emb = self.linear2(nn.silu(condition))
        scale, shift = mx.split(emb, 2, axis=-1)
        return x * (1 + scale[:, None, :]) + shift[:, None, :]


def bicubic_interpolate_pos_encoding(
    pos_embed: mx.array, patch_size_out: int
) -> mx.array:
    """Mirrors Dinov2Embeddings.interpolate_pos_encoding exactly, incl. the
    `+0.1` fudge factor and scale_factor-based (not target-size-based)
    resize. MLX's nn.Upsample(mode="cubic") does NOT match PyTorch's
    bicubic kernel closely enough (see debug_torch_ops.py /
    debug_mlx_ops.py: 0.25 max-abs-diff on a synthetic test), so this uses
    a numpy port of PyTorch's exact cubic-convolution algorithm instead
    (bicubic_numpy.py, verified to 1.2e-6). Only ever called once per model
    load in practice - SF3D always feeds a fixed 512x512 image, so this
    result is identical on every forward pass."""
    num_positions = pos_embed.shape[1] - 1  # 1369 = 37*37
    dim = pos_embed.shape[-1]
    src_grid = int(math.sqrt(num_positions))  # 37

    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]

    h = patch_size_out + 0.1  # matches torch's fudge factor
    scale_factor = h / src_grid

    patch_pos_embed_np = np.array(patch_pos_embed).reshape(1, src_grid, src_grid, dim)
    resized_np = bicubic_resize_nhwc(
        patch_pos_embed_np, patch_size_out, patch_size_out, scale_factor
    )
    patch_pos_embed = mx.array(resized_np).reshape(
        1, patch_size_out * patch_size_out, dim
    )

    return mx.concatenate([class_pos_embed[:, None, :], patch_pos_embed], axis=1)


class Dinov2Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(HIDDEN_SIZE, eps=LAYER_NORM_EPS)
        self.norm1_modulation = Modulation(HIDDEN_SIZE, MODULATION_COND_DIM)
        self.query = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.key = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.value = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.attn_out = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.layer_scale1 = mx.ones((HIDDEN_SIZE,))

        self.norm2 = nn.LayerNorm(HIDDEN_SIZE, eps=LAYER_NORM_EPS)
        self.norm2_modulation = Modulation(HIDDEN_SIZE, MODULATION_COND_DIM)
        self.fc1 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE * MLP_RATIO)
        self.fc2 = nn.Linear(HIDDEN_SIZE * MLP_RATIO, HIDDEN_SIZE)
        self.layer_scale2 = mx.ones((HIDDEN_SIZE,))

    def __call__(self, x: mx.array, modulation_cond: mx.array) -> mx.array:
        b, n, _ = x.shape

        h = self.norm1(x)
        h = self.norm1_modulation(h, modulation_cond)
        q = self.query(h).reshape(b, n, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        k = self.key(h).reshape(b, n, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        v = self.value(h).reshape(b, n, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=HEAD_DIM**-0.5)
        attn = attn.transpose(0, 2, 1, 3).reshape(b, n, HIDDEN_SIZE)
        attn = self.attn_out(attn)
        attn = attn * self.layer_scale1
        x = attn + x

        h = self.norm2(x)
        h = self.norm2_modulation(h, modulation_cond)
        h = self.fc1(h)
        h = nn.gelu(h)
        h = self.fc2(h)
        h = h * self.layer_scale2
        x = h + x
        return x


class Dinov2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token = mx.zeros((1, 1, HIDDEN_SIZE))
        self.patch_conv = nn.Conv2d(
            3, HIDDEN_SIZE, kernel_size=PATCH_SIZE, stride=PATCH_SIZE
        )
        self.position_embeddings = mx.zeros((1, 1370, HIDDEN_SIZE))
        self.layers = [Dinov2Layer() for _ in range(NUM_LAYERS)]
        self.final_norm = nn.LayerNorm(HIDDEN_SIZE, eps=LAYER_NORM_EPS)

    def __call__(
        self, pixel_values_nhwc: mx.array, modulation_cond: mx.array
    ) -> mx.array:
        b, height, width, _ = pixel_values_nhwc.shape
        patches = self.patch_conv(pixel_values_nhwc)  # (B, H/14, W/14, C)
        grid_h, grid_w = patches.shape[1], patches.shape[2]
        patches = patches.reshape(b, grid_h * grid_w, HIDDEN_SIZE)

        cls = mx.broadcast_to(self.cls_token, (b, 1, HIDDEN_SIZE))
        x = mx.concatenate([cls, patches], axis=1)

        assert grid_h == grid_w
        pos = bicubic_interpolate_pos_encoding(self.position_embeddings, grid_h)
        x = x + pos

        for layer in self.layers:
            x = layer(x, modulation_cond)

        return self.final_norm(x)


def load_into(
    model: Dinov2Model, weights: dict, prefix: str = "image_tokenizer.model."
):
    model.cls_token = weights[prefix + "embeddings.cls_token"]
    model.position_embeddings = weights[prefix + "embeddings.position_embeddings"]

    conv_w = weights[
        prefix + "embeddings.patch_embeddings.projection.weight"
    ]  # (O,I,kh,kw)
    model.patch_conv.weight = mx.transpose(
        conv_w, (0, 2, 3, 1)
    )  # -> (O,kh,kw,I) for MLX
    model.patch_conv.bias = weights[
        prefix + "embeddings.patch_embeddings.projection.bias"
    ]

    model.final_norm.weight = weights[prefix + "layernorm.weight"]
    model.final_norm.bias = weights[prefix + "layernorm.bias"]

    for i, layer in enumerate(model.layers):
        p = f"{prefix}encoder.layer.{i}."
        layer.norm1.weight = weights[p + "norm1.weight"]
        layer.norm1.bias = weights[p + "norm1.bias"]
        layer.norm1_modulation.linear2.weight = weights[
            p + "norm1_modulation.linear2.weight"
        ]
        layer.norm1_modulation.linear2.bias = weights[
            p + "norm1_modulation.linear2.bias"
        ]

        layer.query.weight = weights[p + "attention.attention.query.weight"]
        layer.query.bias = weights[p + "attention.attention.query.bias"]
        layer.key.weight = weights[p + "attention.attention.key.weight"]
        layer.key.bias = weights[p + "attention.attention.key.bias"]
        layer.value.weight = weights[p + "attention.attention.value.weight"]
        layer.value.bias = weights[p + "attention.attention.value.bias"]
        layer.attn_out.weight = weights[p + "attention.output.dense.weight"]
        layer.attn_out.bias = weights[p + "attention.output.dense.bias"]
        layer.layer_scale1 = weights[p + "layer_scale1.lambda1"]

        layer.norm2.weight = weights[p + "norm2.weight"]
        layer.norm2.bias = weights[p + "norm2.bias"]
        layer.norm2_modulation.linear2.weight = weights[
            p + "norm2_modulation.linear2.weight"
        ]
        layer.norm2_modulation.linear2.bias = weights[
            p + "norm2_modulation.linear2.bias"
        ]

        layer.fc1.weight = weights[p + "mlp.fc1.weight"]
        layer.fc1.bias = weights[p + "mlp.fc1.bias"]
        layer.fc2.weight = weights[p + "mlp.fc2.weight"]
        layer.fc2.bias = weights[p + "mlp.fc2.bias"]
        layer.layer_scale2 = weights[p + "layer_scale2.lambda1"]
