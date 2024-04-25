from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    head_dim: int
    num_transformer_layers: int
    model_dim: int
    vocab_size: int
    ffn_dim_divisor: int
    num_query_heads: List
    num_kv_heads: List
    ffn_multipliers: List
    ffn_with_glu: bool = True
    normalize_qk_projections: bool = True
    share_input_output_layers: bool = True
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    def __post_init__(self):
        if self.rope_scaling:
            required_keys = {"long_factor", "type"}
            if not all(key in self.rope_scaling for key in required_keys):
                raise ValueError(f"rope_scaling must contain keys {required_keys}")

            if self.rope_scaling["type"] != "linear":
                print(
                    "[WARNING] rope_scaling 'type' currently only supports 'linear' setting rope scaling to false."
                )
                self.rope_scaling = None


def make_divisible(
    v: Union[float, int],
    divisor: Optional[int] = 8,
    min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by the divisor
    It can be seen at:
    https://github.com/tensorflow/models/blob/2cfc99eff5e5eb729c6793d2f3d03aa1c9be2b15/research/slim/nets/mobilenet/mobilenet.py#L62
    Args:
        v: input value
        divisor: default to 8
        min_value: minimum divisor value
    Returns:
        new_v: new divisible value
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.head_dim = head_dim = args.head_dim
        self.layer_id = layer_id
        self.model_dim = model_dim = args.model_dim

        self.n_heads = n_heads = args.num_query_heads[layer_id]
        self.n_kv_heads = n_kv_heads = args.num_kv_heads[layer_id]s
        self.scale = head_dim**-0.5

        op_size = (n_heads + (n_kv_heads * 2)) * head_dim
        self.qkv_proj = nn.Linear(model_dim, op_size, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, model_dim, bias=False)

        self.normalize_qk_projections = args.normalize_qk_projections

        if self.normalize_qk_projections:
            self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
            self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

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

        qkv = self.qkv_proj(x)

        # [B, S, (q_h + k_h + v_h) * h] --> [B, S, (q_h + k_h + v_h), h] -> [B, (q_h + k_h + v_h), S, h]
        qkv = qkv.reshape(
            B, L, self.n_heads + (self.n_kv_heads * 2), self.head_dim
        ).transpose(0, 2, 1, 3)

        # [B, (q_h + k_h + v_h), S, h] --> [B, q_h, S h], [B, k_h, S, h], [B, v_h, S, h]
        queries = qkv[:, : self.n_heads, :, :]
        keys = qkv[:, self.n_heads : (self.n_heads + self.n_kv_heads), :, :]
        values = qkv[:, (self.n_heads + self.n_kv_heads) :, :, :]

        # Prepare the queries, keys and values for the attention computation
        if self.normalize_qk_projections:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

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

        return self.out_proj(output), (keys, values)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.args = args
        dim = args.model_dim
        ffn_multiplier = args.ffn_multipliers[layer_id]

        intermediate_dim = int(
            make_divisible(
                ffn_multiplier * args.model_dim,
                divisor=args.ffn_dim_divisor,
            )
        )

        self.proj_1 = nn.Linear(dim, 2 * intermediate_dim, bias=False)
        self.proj_2 = nn.Linear(intermediate_dim, dim, bias=False)

    def __call__(self, x) -> mx.array:
        x = self.proj_1(x)
        gate, x = mx.split(x, 2, axis=-1)
        return self.proj_2(nn.silu(gate) * x)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        dim = args.model_dim
        self.attn = Attention(args, layer_id=layer_id)
        self.ffn = MLP(args, layer_id=layer_id)
        self.ffn_norm = nn.RMSNorm(dim, eps=args.rms_norm_eps)
        self.attn_norm = nn.RMSNorm(dim, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r, cache = self.attn(self.attn_norm(x), mask, cache)
        h = x + r
        r = self.ffn(self.ffn_norm(h))
        out = h + r
        return out, cache


class OpenELMModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_transformer_layers = args.num_transformer_layers
        assert self.vocab_size > 0
        self.token_embeddings = nn.Embedding(args.vocab_size, args.model_dim)
        self.layers = [
            TransformerBlock(args, layer_id=layer_id)
            for layer_id in range(self.num_transformer_layers)
        ]
        self.norm = nn.RMSNorm(args.model_dim, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.token_embeddings(inputs)

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
        self.transformer = OpenELMModel(args)
        if not args.share_input_output_layers:
            self.lm_head = nn.Linear(args.model_dim, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out, cache = self.transformer(inputs, cache)
        if self.args.share_input_output_layers:
            out = self.transformer.token_embeddings.as_linear(out)
        else:
            out = self.lm_head(out)

        return out, cache

    @property
    def layers(self):
        return self.transformer.layers
