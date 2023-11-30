# Copyright © 2023 Apple Inc.

import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten


class LoRALinear(nn.Module):
    @staticmethod
    def from_linear(linear: nn.Linear, rank: int = 8):
        input_dims, output_dims = linear.weight.shape
        lora_lin = LoRALinear(input_dims, output_dims, rank)
        lora_lin.linear = linear
        return lora_lin

    def __init__(
        self, input_dims: int, output_dims: int, lora_rank: int = 8, bias: bool = False
    ):
        super().__init__()

        # Regular linear layer weights
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)

        # Low rank lora weights
        scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(input_dims, lora_rank),
        )
        self.lora_b = mx.zeros(shape=(lora_rank, output_dims))

    def __call__(self, x):
        y = self.linear(x)
        z = (x @ self.lora_a) @ self.lora_b
        return y + 2.0 * z


class LlamaAttention(nn.Module):
    def __init__(self, dims: int, num_heads: int):
        super().__init__()

        self.num_heads = num_heads

        self.rope = nn.RoPE(dims // num_heads, traditional=True)

        self.query_proj = nn.Linear(dims, dims, bias=False)
        self.key_proj = nn.Linear(dims, dims, bias=False)
        self.value_proj = nn.Linear(dims, dims, bias=False)
        self.out_proj = nn.Linear(dims, dims, bias=False)

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


class LlamaEncoderLayer(nn.Module):
    def __init__(self, dims: int, mlp_dims: int, num_heads: int):
        super().__init__()

        self.attention = LlamaAttention(dims, num_heads)

        self.norm1 = nn.RMSNorm(dims)
        self.norm2 = nn.RMSNorm(dims)

        self.linear1 = nn.Linear(dims, mlp_dims, bias=False)
        self.linear2 = nn.Linear(dims, mlp_dims, bias=False)
        self.linear3 = nn.Linear(mlp_dims, dims, bias=False)

    def __call__(self, x, mask=None, cache=None):
        y = self.norm1(x)
        y, cache = self.attention(y, y, y, mask, cache)
        x = x + y

        y = self.norm2(x)
        a = self.linear1(y)
        b = self.linear2(y)
        y = a * mx.sigmoid(a) * b
        y = self.linear3(y)
        x = x + y

        return x, cache


class Llama(nn.Module):
    def __init__(
        self, num_layers: int, vocab_size: int, dims: int, mlp_dims: int, num_heads: int
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dims)
        self.layers = [
            LlamaEncoderLayer(dims, mlp_dims, num_heads) for _ in range(num_layers)
        ]
        self.norm = nn.RMSNorm(dims)
        self.out_proj = nn.Linear(dims, vocab_size, bias=False)

    def __call__(self, x):
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])

        x = self.embedding(x)

        for l in self.layers:
            x, _ = l(x, mask)

        x = self.norm(x)

        return self.out_proj(x)

    def generate(self, x, temp=1.0):
        cache = []
        try:
            # Make an additive causal mask. We will need that to process the prompt.
            mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
            mask = mask.astype(self.embedding.weight.dtype)

            # First we process the prompt x the same was as in __call__ but
            # save the caches in cache
            x = self.embedding(x)
            for l in self.layers:
                x, c = l(x, mask=mask)
                # We store the per layer cache in a simple python list
                cache.append(c)
            x = self.norm(x)
            # We only care about the last logits that generate the next token
            y = self.out_proj(x[:, -1])
            y = mx.random.categorical(y * (1 / temp))

            # y now has size [1]
            yield y

            # Now we parsed the prompt and generated the first token we
            # need to feed it back into the model and loop to generate the
            # rest.
            while True:
                # Unsqueezing the last dimension to add a sequence length
                # dimension of 1
                x = y[:, None]

                x = self.embedding(x)
                for i in range(len(cache)):
                    # We are overwriting the arrays in the cache list. When
                    # the computation will happen, MLX will be discarding the
                    # old cache the moment it is not needed anymore.
                    x, cache[i] = self.layers[i](x, mask=None, cache=cache[i])
                x = self.norm(x)
                y = self.out_proj(x[:, -1])
                y = mx.random.categorical(y * (1 / temp))

                yield y

        finally:
            del cache


def load_model(model_path):
    weights = mx.load(model_path)
    mlp_dims, dims = weights["layers.0.linear1.weight"].shape
    num_heads = dims // 128
    num_layers = max(int(l.split(".")[1]) for l in weights.keys() if "layers" in l) + 1
    vocab_size = weights["out_proj.weight"].shape[-1]
    model = Llama(num_layers, vocab_size, dims, mlp_dims, num_heads)
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return model
