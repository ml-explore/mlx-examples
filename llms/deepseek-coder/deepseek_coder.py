import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from transformers import AutoTokenizer


@dataclass
class ModelArgs:
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    num_key_value_heads: int = 32
    max_position_embeddings: int = 16384
    rms_norm_eps: float = 1e-6
    intermediate_size: int = 11008
    rope_theta: float = 100000
    rope_scaling_factor: float = 4.0
    vocab_size: int = 32256


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def _norm(self, x):
        return x * mx.rsqrt(x.square().mean(-1, keepdims=True) + self.eps)

    def __call__(self, x):
        output = self._norm(x.astype(mx.float32)).astype(x.dtype)
        return self.weight * output


class LinearScalingRoPE(nn.RoPE):
    def __init__(
        self, dims: int, rope_scaling_factor: float = 4.0, base: float = 10000
    ):
        super().__init__(dims)
        self.base = base
        self.rope_scaling_factor = rope_scaling_factor

    def __call__(self, x, offset: int = 0):
        shape = x.shape
        x = mx.reshape(x, (-1, shape[-2], shape[-1]))
        N = x.shape[1] + offset
        costheta, sintheta = LinearScalingRoPE.create_cos_sin_theta(
            N,
            self.dims,
            offset=offset,
            base=self.base,
            rope_scaling_factor=self.rope_scaling_factor,
            dtype=x.dtype,
        )

        rx = self._compute_rope(costheta, sintheta, x)

        return mx.reshape(rx, shape)

    @staticmethod
    def create_cos_sin_theta(
        N: int,
        D: int,
        offset: int = 0,
        base: float = 10000,
        rope_scaling_factor: float = 1.0,
        dtype=mx.float32,
    ):
        D = D // 2
        positions = mx.arange(offset, N, dtype=dtype)
        positions = positions / rope_scaling_factor
        freqs = mx.exp(-mx.arange(0.0, D, dtype=dtype) * (math.log(base) / D))
        theta = mx.reshape(positions, (-1, 1)) * mx.reshape(freqs, (1, -1))
        return mx.cos(theta), mx.sin(theta)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads: int = args.num_attention_heads
        self.num_key_value_heads: int = args.num_key_value_heads
        self.repeats = self.num_attention_heads // self.num_key_value_heads

        self.head_dim = args.hidden_size // args.num_attention_heads

        self.scale = self.head_dim**-0.5

        self.wq = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(
            args.hidden_size, args.num_key_value_heads * self.head_dim, bias=False
        )
        self.wv = nn.Linear(
            args.hidden_size, args.num_key_value_heads * self.head_dim, bias=False
        )
        self.wo = nn.Linear(
            args.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )
        self.rope = LinearScalingRoPE(
            self.head_dim,
            rope_scaling_factor=args.rope_scaling_factor,
            base=args.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.num_attention_heads, -1).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        def repeat(a):
            a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)
            return a.reshape([B, self.num_attention_heads, L, -1])

        keys, values = map(repeat, (keys, values))

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores += mask
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.wo(output), (keys, values)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w1 = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.w2 = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.w3 = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)

    def __call__(self, x) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ffn_norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r, cache = self.attention(self.attention_norm(x), mask, cache)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        return out, cache


class DeepseekCoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.output = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, x, mask=None, cache=None):
        x = self.tok_embeddings(x)
        mask = None
        T = x.shape[1]
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            mask = mask.astype(x.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            x, cache[e] = layer(x, mask, cache[e])
        x = self.norm(x)
        return self.output(x), cache


def generate(
    prompt: mx.array,
    model: DeepseekCoder,
    temp: float = 0.0,
):
    def sample(logits):
        if temp == 0:
            return mx.argmax(logits, axis=-1)
        else:
            return mx.random.categorical(logits * (1 / temp))

    y = prompt
    cache = None
    while True:
        logits, cache = model(y[None], cache=cache)
        logits = logits[:, -1, :]
        y = sample(logits)
        yield y


def load_model(model_path: str):
    model_path = Path(model_path)
    with open(model_path / "config.json", "r") as f:
        config = json.load(f)
        config.pop("model_type")
        quantization = config.pop("quantization", None)
        model_args = ModelArgs(**config)

    model = DeepseekCoder(model_args)
    weights = mx.load(str(model_path / "weights.npz"))
    if quantization is not None:
        nn.QuantizedLinear.quantize_module(model, **quantization)
    model.update(tree_unflatten(list(weights.items())))

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepseek coder inference script")
    parser.add_argument(
        "--model-path",
        type=str,
        default="mlx_model",
        help="The path to the mlx model weights, tokenizer, and config",
    )
    parser.add_argument(
        "--prompt",
        help="The message to be processed by the model",
        default="### Instruction: \nwrite a quick sort algorithm in python.\n### Response: \n",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp",
        help="The sampling temperature.",
        type=float,
        default=0.6,
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")
    args = parser.parse_args()

    mx.random.seed(args.seed)

    model, tokenizer = load_model(args.model_path)

    prompt = tokenizer(args.prompt, return_tensors="np", return_attention_mask=False,)[
        "input_ids"
    ][0]

    prompt = mx.array(prompt)

    print(args.prompt, end="", flush=True)

    tokens = []
    skip = 0
    for token, _ in zip(
        generate(prompt, model, args.temp),
        range(args.max_tokens),
    ):
        if token == tokenizer.eos_token_id:
            break
        tokens.append(token.item())
        s = tokenizer.decode(tokens)
        print(s[skip:], end="", flush=True)
        skip = len(s)

    print(tokenizer.decode(tokens)[skip:], flush=True)
