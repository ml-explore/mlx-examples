# Copyright © 2023-2024 Apple Inc.

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    attention_bias: bool
    moe_topk: int
    num_experts: int
    num_shared_expert: int
    use_mixed_mlp_moe: bool
    use_qk_norm: bool
    rms_norm_eps: float
    rope_theta: float
    use_cla: bool
    cla_share_factor: 2
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False

    def __post_init__(self):

        if self.rope_scaling:
            required_keys = {"factor", "type"}
            if not all(key in self.rope_scaling for key in required_keys):
                raise ValueError(f"rope_scaling must contain keys {required_keys}")


class DynamicNTKAlphaRoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        base: float = 10000,
        scaling_alpha: float = 1.0,
    ):
        super().__init__()
        self.dims = dims
        base = base * scaling_alpha ** (dims / (dims - 2))
        self._freqs = base ** (mx.arange(0, self.dims, 2) / self.dims)

    def __call__(self, x, offset: int = 0):
        return mx.fast.rope(
            x,
            self.dims,
            traditional=False,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


class Attention(nn.Module):
    def __init__(self, kv_proj: bool, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        assert args.num_key_value_heads is not None
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        head_dim = args.hidden_size // n_heads
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.attention_bias)
        if kv_proj:
            self.k_proj = nn.Linear(
                dim, n_kv_heads * head_dim, bias=args.attention_bias
            )
            self.v_proj = nn.Linear(
                dim, n_kv_heads * head_dim, bias=args.attention_bias
            )
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.attention_bias)
        self.use_qk_norm = args.use_qk_norm
        if self.use_qk_norm:
            self.query_layernorm = nn.RMSNorm(head_dim, args.rms_norm_eps)
            self.key_layernorm = nn.RMSNorm(head_dim, args.rms_norm_eps)

        self.rope = DynamicNTKAlphaRoPE(
            head_dim,
            base=args.rope_theta,
            scaling_alpha=args.rope_scaling["alpha"],
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        kv_states=None,
    ) -> mx.array:
        B, L, D = x.shape

        queries = self.q_proj(x)
        if kv_states is None:
            keys, values = self.k_proj(x), self.v_proj(x)
            kv_states = keys, values
        else:
            keys, values = kv_states

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        offset = cache.offset if cache else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)
        if self.use_qk_norm:
            queries = self.query_layernorm(queries)
            keys = self.key_layernorm(keys)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), kv_states


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Gate(nn.Module):
    def __init__(self, dim, num_experts):
        super().__init__()
        self.wg = nn.Linear(dim, num_experts, bias=False)

    def __call__(self, x) -> mx.array:
        return self.wg(x)


class MoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        intermediate_size = args.intermediate_size
        self.use_shared_mlp = args.use_mixed_mlp_moe

        if args.use_mixed_mlp_moe:
            self.shared_mlp = MLP(dim, intermediate_size * args.num_shared_expert)

        self.num_experts = num_experts = args.num_experts
        self.top_k = args.moe_topk

        self.gate = Gate(dim, num_experts)
        self.switch_mlp = SwitchGLU(dim, intermediate_size, num_experts)

    def __call__(
        self,
        x: mx.array,
    ):
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)

        k = self.top_k
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(gates, inds, axis=-1)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        if self.use_shared_mlp:
            shared_expert_output = self.shared_mlp(x)
            y = y + shared_expert_output

        return y


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, kv_proj: bool):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(kv_proj, args)
        if args.num_experts == 1:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)
        else:
            self.mlp = MoeBlock(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        shared_kv_states: Optional[Tuple[mx.array, mx.array]] = None,
    ):
        r, shared_kv_states = self.self_attn(
            self.input_layernorm(x), mask, cache, shared_kv_states
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, shared_kv_states


class HunYuanModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(
                args=args,
                kv_proj=(not args.use_cla) or (i % args.cla_share_factor) == 0,
            )
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        if mask is None:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            if (not self.args.use_cla) or i % self.args.cla_share_factor == 0:
                shared_kv_states = None
            h, shared_kv_states = layer(h, mask, c, shared_kv_states)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = HunYuanModel(args)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        out = self.model(inputs, mask, cache)
        return self.model.embed_tokens.as_linear(out)

    def sanitize(self, weights):

        if "model.layers.0.mlp.gate_and_up_proj.weight" in weights:
            new_weights = {}
            D = self.args.hidden_size
            n_kv_heads = self.args.num_key_value_heads
            n_kv_groups = self.args.num_attention_heads // n_kv_heads
            head_dim = D // self.args.num_attention_heads
            for k, v in weights.items():
                if "qkv_proj" in k:
                    v = v.reshape(n_kv_heads, n_kv_groups + 2, head_dim, -1)
                    splits = v.split([n_kv_groups, n_kv_groups + 1], axis=1)
                    for k_up, v_new in zip(["q_proj", "k_proj", "v_proj"], splits):
                        k_new = k.replace("qkv_proj", k_up)
                        new_weights[k_new] = mx.flatten(v_new, 0, 2)
                elif "gate_and_up_proj" in k:
                    splits = v.split(2, axis=0)
                    for k_up, v_new in zip(["up_proj", "gate_proj"], splits):
                        k_new = k.replace("gate_and_up_proj", k_up)
                        new_weights[k_new] = v_new
                else:
                    new_weights[k] = v
            weights = new_weights

        if "model.layers.0.mlp.experts.0.up_proj.weight" not in weights:
            return weights
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            for n in ["up_proj", "down_proj", "gate_proj"]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{n}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{n}.{k}")
                            for e in range(self.args.num_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{n}.{k}"] = mx.stack(to_join)
        return weights

    @property
    def layers(self):
        return self.model.layers
