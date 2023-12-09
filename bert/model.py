import numpy as np
from typing import Optional
from dataclasses import dataclass
from transformers import BertTokenizer
from mlx.utils import tree_unflatten

import mlx.core as mx
import mlx.nn as nn
import argparse
import numpy
import math


@dataclass
class ModelArgs:
    intermediate_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    vocab_size: int = 30522
    attention_probs_dropout_prob: float = 0.1
    hidden_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    max_position_embeddings: int = 512


model_configs = {
    "bert-base-uncased": ModelArgs(),
    "bert-base-cased": ModelArgs(),
    "bert-large-uncased": ModelArgs(
        intermediate_size=1024, num_attention_heads=16, num_hidden_layers=24
    ),
    "bert-large-cased": ModelArgs(
        intermediate_size=1024, num_attention_heads=16, num_hidden_layers=24
    ),
}


class MultiHeadAttention(nn.Module):
    """
    Minor update to the MultiHeadAttention module to ensure that the
    projections use bias.
    """

    def __init__(
        self,
        dims: int,
        num_heads: int,
        query_input_dims: Optional[int] = None,
        key_input_dims: Optional[int] = None,
        value_input_dims: Optional[int] = None,
        value_dims: Optional[int] = None,
        value_output_dims: Optional[int] = None,
    ):
        super().__init__()

        if (dims % num_heads) != 0:
            raise ValueError(
                f"The input feature dimensions should be divisible by the number of heads ({dims} % {num_heads}) != 0"
            )

        query_input_dims = query_input_dims or dims
        key_input_dims = key_input_dims or dims
        value_input_dims = value_input_dims or key_input_dims
        value_dims = value_dims or dims
        value_output_dims = value_output_dims or dims

        self.num_heads = num_heads
        self.query_proj = nn.Linear(query_input_dims, dims, True)
        self.key_proj = nn.Linear(key_input_dims, dims, True)
        self.value_proj = nn.Linear(value_input_dims, value_dims, True)
        self.out_proj = nn.Linear(value_dims, value_output_dims, True)

    def __call__(self, queries, keys, values, mask=None):
        queries = self.query_proj(queries)
        keys = self.key_proj(keys)
        values = self.value_proj(values)

        num_heads = self.num_heads
        B, L, D = queries.shape
        _, S, _ = keys.shape
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, num_heads, -1).transpose(0, 2, 3, 1)
        values = values.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)

        # Dimensions are [batch x num heads x sequence x hidden dim]
        scale = math.sqrt(1 / queries.shape[-1])
        scores = (queries * scale) @ keys
        if mask is not None:
            mask = self.convert_mask_to_additive_causal_mask(mask)
            mask = mx.expand_dims(mask, (1, 2))
            mask = mx.broadcast_to(mask, scores.shape)
            scores = scores + mask.astype(scores.dtype)
        scores = mx.softmax(scores, axis=-1)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.out_proj(values_hat)

    def convert_mask_to_additive_causal_mask(
        self, mask: mx.array, dtype: mx.Dtype = mx.float32
    ) -> mx.array:
        mask = mask == 0
        mask = mask.astype(dtype) * -1e9
        return mask


class TransformerEncoderLayer(nn.Module):
    """
    A transformer encoder layer with (the original BERT) post-normalization.
    """

    def __init__(
        self,
        dims: int,
        num_heads: int,
        mlp_dims: Optional[int] = None,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        mlp_dims = mlp_dims or dims * 4
        self.attention = MultiHeadAttention(dims, num_heads)
        self.ln1 = nn.LayerNorm(dims, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(dims, eps=layer_norm_eps)
        self.linear1 = nn.Linear(dims, mlp_dims)
        self.linear2 = nn.Linear(mlp_dims, dims)
        self.gelu = nn.GELU()

    def __call__(self, x, mask):
        attention_out = self.attention(x, x, x, mask)
        add_and_norm = self.ln1(x + attention_out)

        ff = self.linear1(add_and_norm)
        ff_gelu = self.gelu(ff)
        ff_out = self.linear2(ff_gelu)
        x = self.ln2(ff_out + add_and_norm)

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self, num_layers: int, dims: int, num_heads: int, mlp_dims: Optional[int] = None
    ):
        super().__init__()
        self.layers = [
            TransformerEncoderLayer(dims, num_heads, mlp_dims)
            for i in range(num_layers)
        ]

    def __call__(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return x


class BertEmbeddings(nn.Module):
    def __init__(self, config: ModelArgs):
        self.word_embeddings = nn.Embedding(config.vocab_size, config.intermediate_size)
        self.token_type_embeddings = nn.Embedding(2, config.intermediate_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.intermediate_size
        )
        self.norm = nn.LayerNorm(config.intermediate_size, eps=config.layer_norm_eps)

    def __call__(self, input_ids: mx.array, token_type_ids: mx.array) -> mx.array:
        words = self.word_embeddings(input_ids)
        position = self.position_embeddings(
            mx.broadcast_to(mx.arange(input_ids.shape[1]), input_ids.shape)
        )
        token_types = self.token_type_embeddings(token_type_ids)

        embeddings = position + words + token_types
        return self.norm(embeddings)


class Bert(nn.Module):
    def __init__(self, config: ModelArgs):
        self.embeddings = BertEmbeddings(config)
        self.encoder = TransformerEncoder(
            num_layers=config.num_hidden_layers,
            dims=config.intermediate_size,
            num_heads=config.num_attention_heads,
        )
        self.pooler = nn.Linear(config.intermediate_size, config.vocab_size)

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> tuple[mx.array, mx.array]:
        x = self.embeddings(input_ids, token_type_ids)
        y = self.encoder(x, attention_mask)
        return y, mx.tanh(self.pooler(y[:, 0]))


def load_model(bert_model: str, weights_path: str) -> tuple[Bert, BertTokenizer]:
    # load the weights npz
    weights = mx.load(weights_path)
    weights = tree_unflatten(list(weights.items()))
    # create and update the model
    model = Bert(model_configs[bert_model])
    model.update(weights)

    tokenizer = BertTokenizer.from_pretrained(bert_model)

    return model, tokenizer


def run(bert_model: str, mlx_model: str):
    model, tokenizer = load_model(bert_model, mlx_model)

    batch = [
        "This is an example of BERT working on MLX.",
        "A second string",
        "This is another string.",
    ]

    tokens = tokenizer(batch, return_tensors="np", padding=True)
    tokens = {key: mx.array(v) for key, v in tokens.items()}

    mlx_output, mlx_pooled = model(**tokens)
    mlx_output = numpy.array(mlx_output)
    mlx_pooled = numpy.array(mlx_pooled)

    print("MLX BERT:")
    print(mlx_output)

    print("\n\nMLX Pooled:")
    print(mlx_pooled[0, :20])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the BERT model using MLX.")
    parser.add_argument(
        "--bert-model",
        type=str,
        default="bert-base-uncased",
        help="The huggingface name of the BERT model to save.",
    )
    parser.add_argument(
        "--mlx-model",
        type=str,
        default="weights/bert-base-uncased.npz",
        help="The path of the stored MLX BERT weights (npz file).",
    )
    args = parser.parse_args()

    run(args.bert_model, args.mlx_model)
