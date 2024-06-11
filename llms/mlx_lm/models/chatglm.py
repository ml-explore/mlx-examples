from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers import dropout
import numpy as np
import math

from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    hidden_size: int
    kv_channels: int
    num_attention_heads: int
    multi_query_group_num: int
    rope_ratio: int
    ffn_hidden_size: int

    layernorm_epsilon: float

    original_rope: bool
    multi_query_attention: bool
    add_bias_linear: bool
    add_qkv_bias: bool
    apply_query_keys_scaling: bool
    attention_softmax_in_fp32: bool
    attention_dropout: bool
    apply_residual_connection_post_layernorm: bool
    fp32_residual_connection: bool
    rmsnorm: bool
    hidden_dropout: bool

    torch_dtype: str


class CoreAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.apply_query_keys_scaling = args.apply_query_keys_scaling
        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32

        if self.apply_query_keys_scaling:
            self.attention_softmax_in_fp32 = True

        projection_size = args.kv_channels * args.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // args.num_attention_heads
        self.num_attention_heads_per_partition = args.num_attention_heads

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_keys_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.coeff = coeff

        self.attention_dropout = nn.Dropout(args.attention_dropout)

    def __call__(self, queries, keys, values, mask):
        if mask is None and queries.shape[2] == keys.shape[2]:
            context_layer = mx.fast.scaled_dot_product_attention(queries, keys, values, mask=True, scale=1)
        else:
            if mask is not None:
                mask = ~mask
            context_layer = mx.fast.scaled_dot_product_attention(queries, keys, values, mask=True, scale=1).transpose(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
            context_layer = context_layer.reshape(*new_context_layer_shape)

        return context_layer


class SelfAttention(nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """
    def __init__(self, args: ModelArgs):
        super(SelfAttention, self).__init__()

        self.projection_size = args.kv_channels * args.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = self.projection_size // args.num_attention_heads
        self.num_attention_heads_per_partition =  args.num_attention_heads

        self.multi_query_attention = args.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size

        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = args.multi_query_group_num
            self.qkv_hidden_size = self.projection_size + 2 * self.hidden_size_per_attention_head * args.multi_query_group_num

        self.query_key_value = nn.Linear(args.hidden_size, self.qkv_hidden_size, bias=args.add_bias_linear or args.add_qkv_bias)

        self.core_attention = CoreAttention(args)

        # Output
        self.dense = nn.Linear(self.projection_size, args.hidden_size, bias=args.add_bias_linear)

        self.rope = nn.RoPE(
            dims=self.hidden_size_per_attention_head,
            traditional=args.original_rope,
            base=args.rope_ratio,
        )

    # def _allocate_memory(self, inference_max_sequence_len, batch_size):
    #     if self.multi_query_attention:
    #         num_attention_heads = self.num_multi_query_groups_per_partition
    #     else:
    #         num_attention_heads = self.num_attention_heads_per_partition
    #     return np.empty(inference_max_sequence_len, batch_size, num_attention_heads, self.hidden_size_per_attention_head)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None
    ):
        # Attention heads [b, sq, h] --> [b, sq, (np * 3 * hn)]
        mixed_x_layer = self.query_key_value(x)

        if self.multi_query_attention:
            (queries, keys, values) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                ],
                axis=-1,
            )

            queries = queries.view(
                queries.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            keys = keys.view(
                keys.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
            values = values.view(
                values.size()[:-1]
                + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )

        else:
            new_tensor_shape = mixed_x_layer.size()[:-1] + (self.num_attention_heads_per_partition, 3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [b, sq, np, 3 * hn] --> 3 [b, sq, np, hn]
            (queries, keys, values) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # [b, sq, np, hn] -> [b, np, sq, hn]
        queries, keys, values = [k.transpose(1, 2) for k in [queries, keys, values]]

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        if self.multi_query_attention:
            keys = keys.unsqueeze(2)
            keys = keys.expand(-1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1)
            keys = keys.contiguous().view(keys.size()[:1] + (self.num_attention_heads_per_partition,) + keys.size()[3:])
            values = values.unsqueeze(2)
            values = values.expand(-1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1)
            values = values.contiguous().view(values.size()[:1] + (self.num_attention_heads_per_partition,) + values.size()[3:])

        context_layer = self.core_attention(queries, keys, values, mask)

        return self.dense(context_layer)


class MLP(nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, args: ModelArgs):
        super(MLP, self).__init__()

        self.add_bias = args.add_bias_linear

        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = nn.Linear(args.hidden_size, args.ffn_hidden_size * 2, bias=self.add_bias,)

        self.activation_func = nn.

        # Project back to h.
        self.dense_4h_to_h = nn.Linear(args.ffn_hidden_size, args.hidden_size, bias=self.add_bias)

    def __call__(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output


class GLMBlock(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, args: ModelArgs):
        super(GLMBlock, self).__init__()
        self.apply_residual_connection_post_layernorm = args.apply_residual_connection_post_layernorm

        self.fp32_residual_connection = args.fp32_residual_connection

        LayerNormFunc = nn.RMSNorm if args.rmsnorm else nn.LayerNorm
        # Layernorm on the input data.
        self.input_layernorm = LayerNormFunc(args.hidden_size, eps=args.layernorm_epsilon)

        # Self attention.
        self.self_attention = SelfAttention(args)
        self.hidden_dropout = args.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNormFunc(args.hidden_size, eps=args.layernorm_epsilon)

        # MLP
        self.mlp = MLP(args)

    def __call__(
    self,
    x: mx.array,
    mask: Optional[mx.array] = None,
    cache: Optional[Tuple[mx.array, mx.array]] = None,
    ):

        layernorm_output = self.input_layernorm(x)

        attention_output, cache = self.self_attention(
            layernorm_output,
            mask,
            cache
        )

        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = x

        dropout = nn.Dropout(self.hidden_dropout)

        layernorm_input = dropout(attention_output)
        layernorm_input = residual + layernorm_input

        layernorm_output = self.post_attention_layernorm(layernorm_input)

        mlp_output = self.mlp(layernorm_output)

        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = dropout(mlp_output)

        return residual + output
