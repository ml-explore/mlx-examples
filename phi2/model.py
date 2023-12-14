from typing import Optional
from dataclasses import dataclass
from mlx.utils import tree_unflatten, tree_map
from transformers import AutoTokenizer

import mlx.core as mx
import mlx.nn as nn
import math

@dataclass
class ModelArgs:
    max_sequence_length: int = 2048
    num_vocab: int = 51200
    model_dim: int = 2560
    num_heads: int = 32
    num_layers: int = 32
    rotary_dim: int = 32


class RoPEAttention(nn.Module):
    def __init__(self, dims: int, num_heads: int, bias: bool = True):
        super().__init__()

        self.num_heads = num_heads

        self.rope = nn.RoPE(dims // num_heads, traditional=True)
        self.query_proj = nn.Linear(dims, dims, bias=bias)
        self.key_proj = nn.Linear(dims, dims, bias=bias)
        self.value_proj = nn.Linear(dims, dims, bias=bias)
        self.out_proj = nn.Linear(dims, dims, bias=bias)

    def __call__(self, queries, keys, values, mask=None, cache=None):
        queries = self.query_proj(queries)
        keys = self.key_proj(keys)
        values = self.value_proj(values)

        # Extract some shapes
        num_heads = self.num_heads
        B, L, D = queries.shape

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)

        # Add RoPE to the queries and keys and combine them with the cache
        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Finally perform the attention computation
        scale = math.sqrt(1 / queries.shape[-1])
        scores = (queries * scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores = scores + mask

        scores = mx.softmax(scores, axis=-1)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        # Note that we return the keys and values to possibly be used as a cache
        return self.out_proj(values_hat), (keys, values)


class ParallelBlock(nn.Module):
    def __init__(self, dims: int, num_heads: int, mlp_dims: Optional[int] = None):
        super().__init__()
        mlp_dims = mlp_dims or dims * 4
        self.self_attention = RoPEAttention(dims, num_heads, bias=True)
        self.ln = nn.LayerNorm(dims)
        self.fc1 = nn.Linear(dims, mlp_dims)
        self.fc2 = nn.Linear(mlp_dims, dims)
        self.act = nn.GELU(approx="precise")

    def __call__(self, x, mask, cache):
        h = self.ln(x)
        attn_h, cache = self.self_attention(h, h, h, mask, cache)
        ff_h = self.fc2(self.act(self.fc1(h)))
        return attn_h + ff_h + x, cache


class TransformerDecoder(nn.Module):
    def __init__(
        self, num_layers: int, dims: int, num_heads: int, mlp_dims: Optional[int] = None
    ):
        super().__init__()
        self.h = [ParallelBlock(dims, num_heads, mlp_dims) for i in range(num_layers)]

    def __call__(self, x, mask, cache):
        if cache is None:
            cache = [None] * len(self.h)

        for e, layer in enumerate(self.h):
            x, cache[e] = layer(x, mask, cache[e])
        return x, cache


class OutputHead(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        self.ln = nn.LayerNorm(config.model_dim)
        self.linear = nn.Linear(config.model_dim, config.num_vocab)

    def __call__(self, inputs):
        return self.linear(self.ln(inputs))


class Phi2(nn.Module):
    def __init__(self, config: ModelArgs):
        self.wte = nn.Embedding(config.num_vocab, config.model_dim)
        self.transformer = TransformerDecoder(
            num_layers=config.num_layers,
            dims=config.model_dim,
            num_heads=config.num_heads,
        )
        self.lm_head = OutputHead(config)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache: mx.array = None,
    ) -> tuple[mx.array, mx.array]:
        x = self.wte(inputs)

        mask = None
        if x.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
            mask = mask.astype(x.dtype)

        y, cache = self.transformer(x, mask, cache)
        return self.lm_head(y), cache


def generate(prompt: mx.array, model: Phi2, temp: Optional[float] = 0.0):
    def sample(logits):
        if temp == 0:
            return mx.argmax(logits, axis=-1)
        else:
            return mx.random.categorical(logits * (1 / temp))

    logits, cache = model(prompt)
    y = sample(logits[:, -1, :])
    yield y

    while True:
        logits, cache = model(y[:, None], cache=cache)
        y = sample(logits.squeeze(1))
        yield y


if __name__ == "__main__":
    model = Phi2(ModelArgs())

    weights = mx.load("weights/phi-2.npz")
    weights = tree_unflatten(list(weights.items()))
    weights = tree_map(lambda p: mx.array(p, mx.float32), weights)

    model.update(weights)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    prompt = tokenizer("Write a detailed analogy between mathematics and a lighthouse.",
        return_tensors="np",
        return_attention_mask=False,
    )["input_ids"]

    prompt = mx.array(prompt)

    tokens_per_eval = 1
    max_tokens = 100

    tokens = []
    for token, _ in zip(generate(prompt, model), range(max_tokens)):
        tokens.append(token)

        if (len(tokens) % tokens_per_eval) == 0:
            mx.eval(tokens)
            s = tokenizer.decode([t.item() for t in tokens])
            print(s, end="", flush=True)
            tokens = []

