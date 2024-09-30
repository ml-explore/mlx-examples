# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .base import BaseModelArgs, create_attention_mask


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    d_model: int
    ffn_config: dict
    attn_config: dict
    n_layers: int
    n_heads: int


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.n_heads
        self.d_model = args.d_model
        self.head_dim = args.d_model // args.n_heads
        self.num_key_value_heads = args.attn_config["kv_n_heads"]
        self.clip_qkv = args.attn_config["clip_qkv"]
        self.rope_theta = args.attn_config["rope_theta"]

        self.scale = self.head_dim**-0.5

        self.Wqkv = nn.Linear(
            args.d_model,
            (self.num_key_value_heads * 2 + self.num_heads) * self.head_dim,
            bias=False,
        )
        self.out_proj = nn.Linear(args.d_model, args.d_model, bias=False)
        self.rope = nn.RoPE(
            self.head_dim,
            traditional=False,
            base=self.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:

        qkv = self.Wqkv(x)
        qkv = mx.clip(qkv, a_min=-self.clip_qkv, a_max=self.clip_qkv)
        splits = [self.d_model, self.d_model + self.head_dim * self.num_key_value_heads]
        queries, keys, values = mx.split(qkv, splits, axis=-1)

        B, L, D = x.shape

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)


class NormAttnNorm(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.norm_1 = nn.LayerNorm(args.d_model, bias=False)
        self.norm_2 = nn.LayerNorm(args.d_model, bias=False)
        self.attn = Attention(args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        h = self.attn(self.norm_1(x), mask=mask, cache=cache)
        x = h + x
        return x, self.norm_2(x)


class MLP(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.v1 = nn.Linear(d_model, ffn_dim, bias=False)
        self.w1 = nn.Linear(d_model, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, d_model, bias=False)
        self.act_fn = nn.silu

    def __call__(self, x: mx.array) -> mx.array:
        current_hidden_states = self.act_fn(self.w1(x)) * self.v1(x)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class Router(nn.Module):
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.layer = nn.Linear(d_model, num_experts, bias=False)

    def __call__(self, x: mx.array):
        return self.layer(x)


class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.d_model = args.d_model
        self.ffn_dim = args.ffn_config["ffn_hidden_size"]
        self.num_experts = args.ffn_config["moe_num_experts"]
        self.num_experts_per_tok = args.ffn_config["moe_top_k"]

        self.router = Router(self.d_model, self.num_experts)
        self.experts = [
            MLP(self.d_model, self.ffn_dim) for _ in range(self.num_experts)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        ne = self.num_experts_per_tok
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1])

        gates = self.router(x)
        gates = mx.softmax(gates.astype(mx.float32), axis=-1)

        inds = mx.stop_gradient(mx.argpartition(-gates, kth=ne - 1, axis=-1)[:, :ne])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = scores / mx.linalg.norm(scores, ord=1, axis=-1, keepdims=True)
        scores = scores.astype(x.dtype)

        if self.training:
            inds = np.array(inds)
            y = mx.zeros((x.shape[0], ne, x.shape[-1]), x.dtype)
            for e, expert in enumerate(self.experts):
                idx1, idx2 = map(mx.array, np.where(inds == e))
                if idx1.size == 0:
                    continue
                y[idx1, idx2] = expert(x[idx1])

            y = (y * scores[:, :, None]).sum(axis=1)
        else:
            y = []
            for xt, st, it in zip(x, scores, inds.tolist()):
                yt = mx.stack([self.experts[e](xt) for e in it], axis=-1)
                yt = (yt * st).sum(axis=-1)
                y.append(yt)
            y = mx.stack(y, axis=0)

        return y.reshape(orig_shape)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.ffn = SparseMoeBlock(args)
        self.norm_attn_norm = NormAttnNorm(args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r, h = self.norm_attn_norm(x, mask, cache)
        out = self.ffn(h) + r
        return out


class DBRX(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.wte = nn.Embedding(args.vocab_size, args.d_model)
        self.blocks = [DecoderLayer(args=args) for _ in range(args.n_layers)]
        self.norm_f = nn.LayerNorm(args.d_model, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.wte(inputs)

        mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.blocks)

        for layer, c in zip(self.blocks, cache):
            h = layer(h, mask, c)

        return self.norm_f(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.transformer = DBRX(args)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.args = args

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.transformer(inputs, cache)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.transformer.blocks

    def sanitize(self, weights):
        # Split experts into sub matrices
        num_experts = self.args.ffn_config["moe_num_experts"]
        dim = self.args.ffn_config["ffn_hidden_size"]

        pattern = "experts.mlp"
        new_weights = {k: v for k, v in weights.items() if pattern not in k}
        for k, v in weights.items():
            if pattern in k:
                experts = [
                    (k.replace(".mlp", f".{e}") + ".weight", sv)
                    for e, sv in enumerate(mx.split(v, num_experts, axis=0))
                ]
                if k.endswith("w2"):
                    experts = [(s, sv.T) for s, sv in experts]
                new_weights.update(experts)
        return new_weights

    @property
    def head_dim(self):
        return self.args.d_model // self.args.n_heads

    @property
    def n_kv_heads(self):
        return self.args.attn_config["kv_n_heads"]
