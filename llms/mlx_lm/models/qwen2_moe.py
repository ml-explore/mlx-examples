from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_experts_per_tok: int
    num_experts: int
    decoder_sparse_step: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int = None
    rope_theta: float = 1000000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_scaling:
            required_keys = {"factor", "type"}
            if not all(key in self.rope_scaling for key in required_keys):
                raise ValueError(f"rope_scaling must contain keys {required_keys}")

            if self.rope_scaling["type"] != "linear":
                raise ValueError("rope_scaling 'type' currently only supports 'linear'")


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        head_dim = args.hidden_size // n_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        rope_scale = (
            1 / args.rope_scaling["factor"]
            if args.rope_scaling is not None and args.rope_scaling["type"] == "linear"
            else 1
        )
        self.rope = nn.RoPE(
            head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
            scale=rope_scale,
        )

    def __call__(
            self,
            x: mx.array,
            mask: Optional[mx.array] = None,
            cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), (keys, values)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2MoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_dim = args.hidden_size
        self.ffn_dim = args.moe_intermediate_size
        self.num_experts = args.num_experts
        self.num_experts_per_tok = args.num_experts_per_tok

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = [
            MLP(self.hidden_dim, self.ffn_dim) for _ in range(self.num_experts)
        ]

        self.shared_expert = MLP(self.hidden_dim, args.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(self.hidden_dim, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        ne = self.num_experts_per_tok
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1])

        gates = self.gate(x)

        inds = mx.stop_gradient(
            mx.argpartition(-gates, kth=ne, axis=-1)[:, :ne]
        )  # TODO remove it once we figure out how to fine tune TopK in MOE

        scores = mx.softmax(gates.astype(mx.float32), axis=-1).astype(gates.dtype)

        scores = mx.take_along_axis(scores, inds, axis=-1)

        if self.training:
            mx.eval(inds)
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
                yt = mx.concatenate([self.experts[e](xt)[:, None] for e in it], axis=-1)
                yt = (yt * st).sum(axis=-1)
                y.append(yt[None, :])
            y = mx.concatenate(y)

        z = self.shared_expert(x)
        z = mx.sigmoid(self.shared_expert_gate(x)) * z

        y = y + z

        return y.reshape(orig_shape)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = Qwen2MoeBlock(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def __call__(
            self,
            x: mx.array,
            mask: Optional[mx.array] = None,
            cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, cache


class Qwen2MoeModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
            self,
            inputs: mx.array,
            cache=None,
    ):
        h = self.embed_tokens(inputs)

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, mask, cache[e])

        return self.norm(h), cache


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen2MoeModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
            self,
            inputs: mx.array,
            cache=None,
    ):
        out, cache = self.model(inputs, cache)
        return self.lm_head(out), cache

    def sanitize(self, weights):
        if self.args.tie_word_embeddings and "lm_head.weight" not in weights:
            weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
        # Remove unused precomputed rotary freqs
        return {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }

    @property
    def layers(self):
        return self.model.layers
