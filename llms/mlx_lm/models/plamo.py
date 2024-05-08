from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    n_shared_head: int = 8
    rope_theta: float = 10000
    rope_traditional: bool = False


class Attention(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        head_dim = self.hidden_size // config.num_attention_heads

        self.q_num_heads = config.num_attention_heads
        self.qk_dim = self.v_dim = head_dim
        self.k_num_heads = self.v_num_heads = int(
            np.ceil(self.q_num_heads / config.n_shared_head)
        )

        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size, self.q_num_heads * self.qk_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.k_num_heads * self.qk_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.v_num_heads * self.v_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.q_num_heads * self.v_dim, self.hidden_size, bias=False
        )
        self.rotary_emb = nn.RoPE(
            head_dim,
            traditional=config.rope_traditional,
            base=config.rope_theta,
            scale=1.0,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        bsz, q_len, _ = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(bsz, q_len, self.q_num_heads, self.qk_dim).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(bsz, q_len, self.k_num_heads, self.qk_dim).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(bsz, q_len, self.v_num_heads, self.v_dim).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            queries = self.rotary_emb(queries, offset=cache.offset)
            keys = self.rotary_emb(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rotary_emb(queries)
            keys = self.rotary_emb(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.scale,
            mask=attention_mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(bsz, q_len, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))  # type: ignore


class PlamoDecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[Any, ...]:
        # from LlamaDecoder
        residual = hidden_states

        hidden_states = self.norm(hidden_states)

        # Self Attention
        hidden_states_sa = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            cache=cache,
        )

        # Fully Connected
        hidden_states_mlp = self.mlp(hidden_states)

        hidden_states = residual + hidden_states_sa + hidden_states_mlp
        return hidden_states


class PlamoDecoder(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.layers = [
            PlamoDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]


class PlamoModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = PlamoDecoder(config)  # type: ignore
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Union[Tuple[mx.array, mx.array], None]]] = None,
    ) -> Tuple[mx.array, Optional[List[Union[Tuple[mx.array, mx.array], None]]]]:
        h = self.embed_tokens(inputs)

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(self.embed_tokens.weight.dtype)

        if cache is None:
            cache = [None for _ in range(len(self.layers.layers))]

        for layer, c in zip(self.layers.layers, cache):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.model_type = args.model_type
        self.model = PlamoModel(args)
        self.lm_head: nn.Module = nn.Linear(
            args.hidden_size, args.vocab_size, bias=False
        )
        self.args = args

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ) -> Tuple[mx.array, mx.array]:
        out = self.model(inputs, cache)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers.layers

    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.args.num_attention_heads // self.args.n_shared_head
