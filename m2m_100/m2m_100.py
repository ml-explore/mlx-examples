# Copyright © 2023 Apple Inc.

import argparse
import glob
import json
import time
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from sentencepiece import SentencePieceProcessor

@dataclass
class ModelArgs:
    vocab_size: int
    max_position_embeddings: int
    encoder_layers: int
    encoder_ffn_dim: int
    encoder_attention_heads: int
    decoder_layers: int
    decoder_ffn_dim: int
    decoder_attention_heads: int
    encoder_layerdrop: float
    decoder_layerdrop: float
    use_cache: bool
    is_encoder_decoder: bool    
    activation_function: str
    d_model: int
    dropout: float
    attention_dropout: float
    activation_dropout: float
    init_std: float
    decoder_start_token_id: int
    scale_embedding: bool
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int

class M2M100Attention(nn.Module):
    def __init__(self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        args: ModelArgs = None):

        super().__init__()
        self.args = args

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads

        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
    
    def _shape(self, tensor: mx.array, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def __call__(self,
        hidden_states: mx.array,
        key_value_states: Optional[mx.array] = None,
        past_key_value: Optional[Tuple[mx.array]] = None,
        attention_mask: Optional[mx.array] = None,
        layer_head_mask: Optional[mx.array] = None,
        ):

        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)

        if (is_cross_attention and past_key_value is not None and past_key_value[0].shape[2] == key_value_states.shape[1]):
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = mx.concatenate([past_key_value[0], key_states], axis=2)
            value_states = mx.concatenate([past_key_value[1], value_states], axis=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        
        if self.is_decoder:
            past_key_value = (key_states, value_states)
    
        proj_shape = (bsz * self.num_heads, tgt_len, self.head_dim)

        query_states = self._shape(query_states, tgt_len, bsz).reshape(proj_shape)
        key_states = key_states.reshape(proj_shape)
        value_states = value_states.reshape(proj_shape)

        src_len = key_states.shape[1]
        attn_weights = query_states @ key_states.transpose(1, 2)

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )
        
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights.reshape(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.reshape(bsz * self.num_heads, tgt_len, src_len)
        
        attn_weights = mx.softmax(attn_weights, axis=-1).astype(attn_weights.dtype)

        attn_outputs = attn_weights @ value_states

        attn_output = attn_outputs.reshape(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output
            

class M2M100EncoderLayer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        self.embed_dim = config.d_model
        self.self_attn = M2M100Attention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
            bias=True,
            is_causal=False,
            args=config
        )

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        if config.activation_function == "gelu":
            self.activation_fn = nn.GELU()
        elif config.activation_function == "relu":
            self.activation_fn = nn.ReLU()
        
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def __call__(self, hidden_states: mx.array, attention_mask: mx.array):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states
        
class M2M100DecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        self.embed_dim = config.d_model
        self.self_attn = M2M100Attention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            args=config
        )

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = M2M100Attention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            args=config
        )

        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        if config.activation_function == "gelu":
            self.activation_fn = nn.GELU()
        elif config.activation_function == "relu":
            self.activation_fn = nn.ReLU()
        
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def __call__(self, hidden_states: mx.array, encoder_hidden_states: mx.array, attention_mask: mx.array, encoder_attention_mask: mx.array):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=None,
        )
        hidden_states = hidden_states + residual

        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            hidden_states = self.encoder_attn(
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_value=None,
            )
            hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)

        hidden_states = hidden_states + residual
        return hidden_states

class M2M100Encoder(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim)
        self.embed_positions = nn.SinusoidalPositionalEncoding(embed_dim)

        self.layers = [M2M100EncoderLayer(config) for _ in range(config.encoder_layers)]
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def __call__(self, input_ids: mx.array, attention_mask: mx.array):

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(input_ids)

        hidden_states = inputs_embeds + embed_pos

        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, 1, 1, input_shape[-1])
        
        for layer in self.layers:
            layer_output = layer(hidden_states, attention_mask)
            hidden_states = layer_output[0]
        
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class M2M100Decoder(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim)
        self.embed_positions = nn.SinusoidalPositionalEncoding(embed_dim)

        self.layers = [M2M100DecoderLayer(config) for _ in range(config.decoder_layers)]
        self.layer_norm = nn.LayerNorm(embed_dim)

    def __call__(self, input_ids: mx.array, encoder_hidden_states: mx.array, attention_mask: mx.array, encoder_attention_mask: mx.array, past_key_values: mx.array):
        
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_ids, past_key_values)

        hidden_states = inputs_embeds + embed_pos

        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, 1, 1, input_shape[-1])
        
        for layer in self.layers:
            layer_output = layer(hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask)
            hidden_states = layer_output[0]   

        hidden_states = self.layer_norm(hidden_states)

        return hidden_states     
        

class M2M100Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        self.config = config
        self.encoder = M2M100Encoder(config)
        self.decoder = M2M100Decoder(config)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array, decoder_input_ids: mx.array, encoder_attention_mask: mx.array, decoder_attention_mask: mx.array, past_key_values: mx.array):

        encoder_hidden_states = self.encoder(input_ids, attention_mask)
        decoder_hidden_states = self.decoder(decoder_input_ids, encoder_hidden_states, decoder_attention_mask, encoder_attention_mask, past_key_values)

        return decoder_hidden_states

class M2M100ForConditionalGeneration(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        self.config = config
        self.model = M2M100Model(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array, decoder_input_ids: mx.array, encoder_attention_mask: mx.array, decoder_attention_mask: mx.array, past_key_values: mx.array):

        decoder_hidden_states = self.model(input_ids, attention_mask, decoder_input_ids, encoder_attention_mask, decoder_attention_mask, past_key_values)
        lm_logits = self.lm_head(decoder_hidden_states)

        return decoder_hidden_states
