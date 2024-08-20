import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, KVCache, create_attention_mask
from .su_rope import SuScaledRotaryEmbedding


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "phimoe"
    vocab_size: int = 30000
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    pad_token_id: Optional[int] = None
    rope_traditional: bool = False
    num_local_experts: int = 8
    num_experts_per_tok: int = 2
    attention_bias: bool = False
    rope_theta: float = 10000.0

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_scaling:
            required_keys = {"long_factor", "type"}
            if not all(key in self.rope_scaling for key in required_keys):
                raise ValueError(f"rope_scaling must contain keys {required_keys}")

            if self.rope_scaling["type"] not in ["longrope", "su", "linear"]:
                print(
                    "[WARNING] rope_scaling 'type' currently only supports 'linear', 'su', and 'longrope'; setting rope scaling to false."
                )
                self.rope_scaling = None


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        assert args.num_key_value_heads is not None
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        head_dim = args.hidden_size // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        rope_scale = 1.0
        if args.rope_scaling and args.rope_scaling["type"] in ["longrope", "su"]:
            self.rope = SuScaledRotaryEmbedding(
                head_dim,
                traditional=False,
                base=args.rope_theta,
                scale=rope_scale,
                max_position_embeddings=args.max_position_embeddings,
                original_max_position_embeddings=args.original_max_position_embeddings,
                short_factor=args.rope_scaling["short_factor"],
                long_factor=args.rope_scaling["long_factor"],
            )
        else:
            if args.rope_scaling and args.rope_scaling["type"] == "linear":
                assert isinstance(args.rope_scaling["factor"], float)
                rope_scale = 1 / args.rope_scaling["factor"]
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
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

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
        return self.o_proj(output)


class PhiMoEBlockSparseTop2MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.ffn_dim = args.intermediate_size
        self.hidden_dim = args.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = nn.GELU()

    def __call__(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(
            hidden_states
        )
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class PhiMoESparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_dim = args.hidden_size
        self.ffn_dim = args.intermediate_size
        self.num_experts = args.num_local_experts
        self.top_k = args.num_experts_per_tok

        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = [PhiMoEBlockSparseTop2MLP(args) for _ in range(self.num_experts)]

    def __call__(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)

        router_logits = self.gate(hidden_states)
        routing_weights = mx.softmax(router_logits, axis=-1)
        expert_indices = mx.argmax(routing_weights, axis=-1)

        final_hidden_states = mx.zeros((batch_size * sequence_length, hidden_dim))

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            expert_mask = expert_indices == expert_idx
            if mx.sum(expert_mask) > 0:
                expert_input = hidden_states[expert_mask]
                expert_output = expert_layer(expert_input)
                final_hidden_states = mx.where(
                    expert_mask[:, None], expert_output, final_hidden_states
                )

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states, router_logits


class PhiMoEDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size

        self.self_attn = Attention(args)
        self.block_sparse_moe = PhiMoESparseMoeBlock(args)
        self.input_layernorm = nn.LayerNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, hidden_states, attention_mask=None, position_ids=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, router_logits


class PhiMoEModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.padding_idx = args.pad_token_id
        self.vocab_size = args.vocab_size

        self.embed_tokens = nn.Embedding(
            args.vocab_size, args.hidden_size, self.padding_idx
        )
        self.layers = [PhiMoEDecoderLayer(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.LayerNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, input_ids, attention_mask=None, position_ids=None):
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states, _ = layer(hidden_states, attention_mask, position_ids)

        hidden_states = self.norm(hidden_states)

        return hidden_states


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model = PhiMoEModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, input_ids, attention_mask=None, position_ids=None):
        hidden_states = self.model(input_ids, attention_mask, position_ids)
        logits = self.lm_head(hidden_states)
        return logits

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads

    def sanitize(self, weights):
        # Remove unused precomputed rotary freqs
        return {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
