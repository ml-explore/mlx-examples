# Copyright © 2024 Apple Inc.

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear

from .layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)


@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux(nn.Module):
    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {params.axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
        )
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
            if params.guidance_embed
            else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = [
            DoubleStreamBlock(
                self.hidden_size,
                self.num_heads,
                mlp_ratio=params.mlp_ratio,
                qkv_bias=params.qkv_bias,
            )
            for _ in range(params.depth)
        ]

        self.single_blocks = [
            SingleStreamBlock(
                self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio
            )
            for _ in range(params.depth_single_blocks)
        ]

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def sanitize(self, weights):
        new_weights = {}
        for k, w in weights.items():
            if k.startswith("model.diffusion_model."):
                k = k[22:]
            if k.endswith(".scale"):
                k = k[:-6] + ".weight"
            for seq in ["img_mlp", "txt_mlp", "adaLN_modulation"]:
                if f".{seq}." in k:
                    k = k.replace(f".{seq}.", f".{seq}.layers.")
                    break
            new_weights[k] = w
        return new_weights

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        if N == 1:
            return

        for block in self.double_blocks:
            block.num_heads //= N
            block.img_attn.num_heads //= N
            block.txt_attn.num_heads //= N
            block.sharding_group = group
            block.img_attn.qkv = shard_linear(
                block.img_attn.qkv, "all-to-sharded", segments=3, group=group
            )
            block.txt_attn.qkv = shard_linear(
                block.txt_attn.qkv, "all-to-sharded", segments=3, group=group
            )
            shard_inplace(block.img_attn.proj, "sharded-to-all", group=group)
            shard_inplace(block.txt_attn.proj, "sharded-to-all", group=group)
            block.img_mlp.layers[0] = shard_linear(
                block.img_mlp.layers[0], "all-to-sharded", group=group
            )
            block.txt_mlp.layers[0] = shard_linear(
                block.txt_mlp.layers[0], "all-to-sharded", group=group
            )
            shard_inplace(block.img_mlp.layers[2], "sharded-to-all", group=group)
            shard_inplace(block.txt_mlp.layers[2], "sharded-to-all", group=group)

        for block in self.single_blocks:
            block.num_heads //= N
            block.hidden_size //= N
            block.linear1 = shard_linear(
                block.linear1,
                "all-to-sharded",
                segments=[1 / 7, 2 / 7, 3 / 7],
                group=group,
            )
            block.linear2 = shard_linear(
                block.linear2, "sharded-to-all", segments=[1 / 5], group=group
            )

    def __call__(
        self,
        img: mx.array,
        img_ids: mx.array,
        txt: mx.array,
        txt_ids: mx.array,
        timesteps: mx.array,
        y: mx.array,
        guidance: Optional[mx.array] = None,
    ) -> mx.array:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = mx.concatenate([txt_ids, img_ids], axis=1)
        pe = self.pe_embedder(ids).astype(img.dtype)

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = mx.concatenate([txt, img], axis=1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)

        return img
