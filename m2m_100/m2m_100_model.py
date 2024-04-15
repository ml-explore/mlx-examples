# Copyright © 2023 Apple Inc.

import os
import time
import argparse
import math
from typing import Optional, Tuple, List

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.sample_utils import top_p_sampling

import torch
import transformers
import numpy as np
from transformers import M2M100Config
from transformers import PreTrainedTokenizerBase, NllbTokenizer, M2M100Config
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from flores200_codes import FLORES_CODES


def _prepare_4d_attention_mask(mask: mx.array, dtype: mx.Dtype, tgt_len: Optional[int] = None):
    """Create 4d attention mask from a 2d mask."""

    bsz, src_len = mask.shape

    if tgt_len is None:
        tgt_len = src_len

    expanded_mask = mask[:, None, None, :]
    expanded_mask = np.broadcast_to(expanded_mask, (bsz, 1, tgt_len, src_len))
    inverted_mask = 1.0 - expanded_mask
    inverted_mask = np.array(inverted_mask)
    
    inverted_mask = np.ma.array(data=inverted_mask, mask = inverted_mask)
    inverted_mask = inverted_mask.filled(fill_value=-np.inf)
    return mx.array(inverted_mask)


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.
    """
    mask = (input_ids != padding_idx).astype(mx.int16)

    incremental_indices = (mx.cumsum(mask, axis=1).astype(input_ids.dtype) + past_key_values_length) * mask
    return incremental_indices + padding_idx



class M2M100SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.num_positions = num_positions
        self.weight = self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)
        self.weights = nn.SinusoidalPositionalEncoding(embedding_dim)
  
    def __call__(self, input_ids: mx.array, past_key_values_length: int = 0):
        bsz, seq_len = input_ids.shape
        position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)

        position_ids = position_ids.reshape(-1)
        output = mx.take(self.weight, position_ids, 0).reshape(bsz, seq_len, self.weight.shape[-1])
        return output
    
    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        return emb_weights
    
    def get_embedding(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)

        emb = mx.exp(mx.arange(half_dim) * -emb)
        emb = mx.expand_dims(mx.arange(num_embeddings),1) * mx.expand_dims(emb, 0)
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=1).reshape(num_embeddings, -1)

        if embedding_dim % 2 == 1:
            emb = mx.concatenate([emb, mx.zeros(num_embeddings, 1)], axis=1)
        
        if padding_idx is not None:
            emb[padding_idx] = 0
        
        return emb


class M2M100Attention(nn.Module):
    def __init__(self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        args: M2M100Config = None):

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
        return tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(self,
        hidden_states: mx.array,
        key_value_states: Optional[mx.array] = None,
        past_key_value: Optional[Tuple[mx.array]] = None,
        attention_mask: Optional[mx.array] = None,
        layer_head_mask: Optional[mx.array] = None,
        ):

        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states) * self.scaling

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
    
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)

        query_states = self._shape(query_states, tgt_len, bsz).reshape(proj_shape)
        key_states = key_states.reshape(proj_shape)
        value_states = value_states.reshape(proj_shape)

        src_len = key_states.shape[1]
        attn_weights = query_states @ key_states.transpose(0, 2, 1)

        if attn_weights.shape != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.shape}"
            )
        
        if attention_mask is not None:
            if attention_mask.shape != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.shape}"
                )

            attn_weights = attn_weights.reshape(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.reshape(bsz * self.num_heads, tgt_len, src_len)
        
        attn_weights = mx.softmax(attn_weights, axis=-1).astype(attn_weights.dtype)

        attn_outputs = attn_weights @ value_states

        attn_output = attn_outputs.reshape(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(0, 2, 1, 3)

        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output
            

class M2M100EncoderLayer(nn.Module):
    def __init__(self, config: M2M100Config):
        super().__init__()

        self.embed_dim = config.d_model
        self.self_attn = M2M100Attention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
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
    def __init__(self, config: M2M100Config):
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

    def __call__(self, hidden_states: mx.array, attention_mask: mx.array, encoder_hidden_states: mx.array, encoder_attention_mask: mx.array, past_key_value: Optional[Tuple[mx.array]] = None):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=self_attn_past_key_value,
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
    def __init__(self, config: M2M100Config, embed_tokens: Optional[nn.Embedding]):
        super().__init__()

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim) if embed_tokens is None else embed_tokens
        self.embed_positions = M2M100SinusoidalPositionalEmbedding(config.max_position_embeddings, embed_dim, self.padding_idx)

        self.layers = [M2M100EncoderLayer(config) for _ in range(config.encoder_layers)]
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def __call__(self, input_ids: mx.array, attention_mask: mx.array):

        input_shape = input_ids.shape
        input_ids = input_ids.reshape(-1, input_shape[-1])

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_ids)

        hidden_states = inputs_embeds + embed_pos

        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask(attention_mask, mx.float16)

        
        for _, layer in enumerate(self.layers):
            layer_output = layer(hidden_states, attention_mask)
            hidden_states = layer_output
        
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class M2M100Decoder(nn.Module):
    def __init__(self, config: M2M100Config, embed_tokens: Optional[nn.Embedding]):
        super().__init__()

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim) if embed_tokens is None else embed_tokens
        self.embed_positions = M2M100SinusoidalPositionalEmbedding(config.max_position_embeddings, embed_dim, self.padding_idx)

        self.layers = [M2M100DecoderLayer(config) for _ in range(config.decoder_layers)]
        self.layer_norm = nn.LayerNorm(embed_dim)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array, encoder_hidden_states: mx.array, encoder_attention_mask: mx.array, past_key_values: mx.array):
        
        input_shape = input_ids.shape
        input_ids = input_ids.reshape(-1, input_shape[-1])

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = _prepare_4d_causal_attention_mask(
            torch.tensor(np.array(attention_mask)), input_shape, torch.tensor(np.array(inputs_embeds)), past_key_values_length
        )
        combined_attention_mask = mx.array(combined_attention_mask.numpy())

        if attention_mask is not None:
            encoder_attention_mask = _prepare_4d_attention_mask(encoder_attention_mask, mx.float16, tgt_len=input_shape[-1])

        embed_pos = self.embed_positions(input_ids)

        hidden_states = inputs_embeds + embed_pos
        
        for layer in self.layers:
            layer_output = layer(hidden_states, combined_attention_mask, encoder_hidden_states, encoder_attention_mask)
            hidden_states = layer_output

        hidden_states = self.layer_norm(hidden_states)

        return hidden_states
        

class M2M100Model(nn.Module):
    def __init__(self, config: M2M100Config):
        super().__init__()

        self.config = config

        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = M2M100Encoder(config, self.shared)
        self.decoder = M2M100Decoder(config, self.shared)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array, decoder_input_ids: mx.array, encoder_attention_mask: mx.array, decoder_attention_mask: mx.array, past_key_values: mx.array):

        encoder_hidden_states = self.encoder(input_ids, attention_mask)
        decoder_hidden_states = self.decoder(decoder_input_ids, encoder_hidden_states, decoder_attention_mask, encoder_attention_mask, past_key_values)

        return decoder_hidden_states

class M2M100ForConditionalGeneration(nn.Module):
    def __init__(self, config: M2M100Config):
        super().__init__()

        self.config = config
        self.model = M2M100Model(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def encode(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        return self.model.encoder(input_ids, attention_mask)
    
    def decode(self, decoder_input_ids, encoder_hidden_states, attention_mask, encoder_attention_mask, past_key_values):
        return self.model.decoder(decoder_input_ids, encoder_hidden_states, attention_mask, encoder_attention_mask, past_key_values)

def convert(model_name: str, mlx_model: str) -> None:
    """
    Convert a model from Hugging Face to MLX format.

    Args:
        model_name (str): Hugging Face model name
        mlx_model (str): Path to save the MLX model
    """
    model = transformers.M2M100ForConditionalGeneration.from_pretrained(model_name)
    # save the tensors
    tensors = {
        key: tensor.numpy() for key, tensor in model.state_dict().items()
    }
    np.savez(mlx_model, **tensors)


def load_mlx_nllb_model(
    model_name: str, weights_path: str, source_language: str, target_language: str
) -> Tuple[M2M100ForConditionalGeneration, PreTrainedTokenizerBase, int]:
    """
    Load a nllb model and tokenizer from the given model name and weights path.
    """
    if source_language in FLORES_CODES and target_language in FLORES_CODES:
        source_language = FLORES_CODES[source_language]
        target_language = FLORES_CODES[target_language]

    if os.path.exists(weights_path):
        mlx_model = weights_path
    else:
        convert(model_name, weights_path)
        assert os.path.exists(weights_path), f"Model checkpoint not found at {weights_path}"

    config = M2M100Config.from_pretrained(model_name)

    model = M2M100ForConditionalGeneration(config)

    #setting to False because of the Positional Encoding layer
    model.load_weights(weights_path, strict=False)

    tokenizer = NllbTokenizer.from_pretrained(model_name,
        src_lang=source_language,
        tgt_lang=target_language
        )
    
    tgt_token_id =  tokenizer.convert_tokens_to_ids(target_language)
    return model, tokenizer, tgt_token_id

def sample(logits: mx.array, temp: float, top_p: float) -> Tuple[mx.array, float]:
    """
    Sample a token from the logits.

    Args:
        logits (mx.array): Logits from the model
        temp (float): Temperature for sampling
        top_p (float): Top-p sampling value

    Returns:
        Tuple[mx.array, float]: Tuple of the token and its probability
    """
    softmax_logits = mx.softmax(logits)

    if temp == 0:
        token = mx.argmax(logits, axis=-1)
    else:
        if top_p > 0 and top_p < 1.0:
            token = top_p_sampling(logits, top_p, temp)
        else:
            token = mx.random.categorical(logits * (1 / temp))

    prob = softmax_logits[0, token]
    return token, prob


def run_translation_mlx(
    model: M2M100ForConditionalGeneration,
    tokenizer: PreTrainedTokenizerBase,
    input_sentences: List[str],
    target_language_token: int,
    max_generation_tokens: int,
    temp: float = 0.0,
    top_p: float = 1.0,
    verbose: bool = False
    ) -> List[str]:

    """
    Run Translation using the MLX model.

    Returns:
         List[str]: List of translated sentences
    """

    bsz = len(input_sentences)
    tokens = tokenizer(input_sentences, return_tensors="np", padding="longest")

    print(tokens['input_ids'].shape)
    print(tokens['input_ids'])
    tokens = {key: mx.array(v) for key, v in tokens.items()}

    decoder_input_ids = mx.array([[model.config.eos_token_id, target_language_token]]*bsz)
    decoder_input_mask = mx.array([[1, 1]]*bsz)

    encoder_tokens = model.encode(tokens["input_ids"], tokens["attention_mask"])

    start_time = time.time()  # Start measuring time

    for _ in range(max_generation_tokens):
        outputs = model.decode(decoder_input_ids,decoder_input_mask, encoder_tokens, tokens["attention_mask"], None)
        logits = model.lm_head(outputs)[:,-1,:]

        next_token, _ = sample(logits, temp, top_p)

        decoder_input_ids = mx.concatenate([decoder_input_ids, next_token.reshape(-1, 1)], axis=1)
        decoder_input_mask = mx.concatenate([decoder_input_mask, mx.ones((bsz,1))], axis=1)

        if next_token[0] == model.config.eos_token_id:
            break


    end_time = time.time()  # Stop measuring time

    if verbose:
        total_tokens = decoder_input_ids.size - (bsz*2)
        total_time = end_time - start_time
        token_per_sec = total_tokens / total_time
        print(f"Generated {total_tokens} tokens in {total_time} seconds.")
        print(f"Token/s: {token_per_sec}")

    translated_sentences = tokenizer.batch_decode(np.array(decoder_input_ids), skip_special_tokens=True)

    return translated_sentences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert M2M style model weights to MLX.")
    parser.add_argument(
        "--nllb-model",
        type=str,
        default="facebook/nllb-200-1.3B",
        help="The huggingface name of the NLLB model to save",
    )
    parser.add_argument(
        "--mlx-model",
        type=str,
        default="weights/nllb-200-1.3B.npz",
        help="The output path for the MLX weights.",
    )
    parser.add_argument("--source_language", type=str, required=True)
    parser.add_argument("--target_language", type=str, required=True)
    parser.add_argument("--input_sentence",
        type=str,
        default="Hello there! Today, we are going to talk about something called the \"discount rate.\"")
    args = parser.parse_args()

    current_directory = os.getcwd()
    weights_path = os.path.join(current_directory, "weights", args.model_name.replace("/", "-") + ".npz")
    model, tokenizer, tgt_token_id = load_mlx_nllb_model(
        model_name=args.model_name,
        weights_path=weights_path,
        source_language=args.source_language,
        target_language=args.target_language
    )

    translated_sentence = run_translation_mlx(
        model=model,
        tokenizer=tokenizer,
        input_sentences=[args.input_sentence],
        target_language_token=tgt_token_id,
        max_generation_tokens=15,
        verbose=True
    )

    print(f"Translated Sentence: {translated_sentence}")
