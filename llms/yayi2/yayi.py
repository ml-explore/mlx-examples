import argparse
import json
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from transformers import AutoTokenizer


@dataclass
class ModelArgs:
    hidden_size: int = 7168
    num_attention_heads: int = 64
    num_hidden_layers: int = 64
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    intermediate_size: int = 16384
    rope_theta: float = 100000
    vocab_size: int = 81920
    rope_traditional: bool = False


class KVLinear(nn.Module):
    def __init__(self, input_dims: int, output_dims: int):
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(output_dims, input_dims),
        )

    def __call__(self, x):
        k, v = mx.split(x @ self.weight.T, 2, axis=-1)
        return k, v


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


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads: int = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.scale = self.head_dim**-0.5

        self.wq = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False
        )
        self.wkwv = KVLinear(args.hidden_size, int(self.head_dim) * 2)

        self.wo = nn.Linear(
            args.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )
        self.rope = nn.RoPE(
            self.head_dim, traditional=args.rope_traditional, base=args.rope_theta
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        q = self.wq(x)
        k, v = self.wkwv(x)

        q = q.reshape(B, L, self.num_attention_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        k = k.reshape(B, L, 1, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, 1, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            k_cache, v_cache = cache
            q = self.rope(q, offset=k_cache.shape[2])
            k = self.rope(k, offset=k_cache.shape[2])
            k = mx.concatenate([k_cache, k], axis=2)
            v = mx.concatenate([v_cache, v], axis=2)

        else:
            q = self.rope(q)
            k = self.rope(k)

        scores = (q * self.scale) @ k.transpose(0, 1, 3, 2)

        if mask is not None:
            scores = scores + mask

        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        v_hat = (scores @ v).transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size)

        return self.wo(v_hat), (k, v)


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


class Yayi(nn.Module):
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
    model: Yayi,
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

    model = Yayi(model_args)
    weights = mx.load(str(model_path / "weights.npz"))
    if quantization is not None:
        nn.QuantizedLinear.quantize_module(model, **quantization)
    parameteres = tree_unflatten(list(weights.items()))

    model.update(parameteres)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yayi inference script")
    parser.add_argument(
        "--model-path",
        type=str,
        default="mlx_model",
        help="The path to the mlx model weights, tokenizer, and config",
    )
    parser.add_argument(
        "--prompt",
        help="The message to be processed by the model",
        default="The winter in Beijing is ",
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

    prompt = tokenizer(
        args.prompt,
        return_tensors="np",
        return_attention_mask=False,
    )[
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
