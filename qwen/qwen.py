# The architecture of qwen is similar to Llama.
# This inference script is mainly for compatibility with the huggingface model of qwen.

import argparse
import mlx.core as mx
import mlx.nn as nn

from mlx.utils import tree_unflatten
from dataclasses import dataclass
from transformers import AutoTokenizer


@dataclass
class ModelArgs:
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    kv_channels: int = 128

    max_position_embeddings: int = 8192
    layer_norm_epsilon: float = 1e-6
    intermediate_size: int = 11008

    no_bias: bool = True

    vocab_size: int = 151936


class QWenAttntion(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.hidden_size = args.hidden_size
        self.num_attention_heads = args.num_attention_heads

        self.hidden_size_per_attention_head = (
            self.hidden_size // self.num_attention_heads
        )

        self.rotary_emb = nn.RoPE(
            self.hidden_size_per_attention_head, traditional=False
        )

        self.proj_size = args.kv_channels * self.num_attention_heads

        self.c_attn = nn.Linear(
            self.hidden_size, self.proj_size * 3, bias=True)
        self.c_proj = nn.Linear(
            self.hidden_size, self.proj_size, bias=not args.no_bias)

        self.scale = self.hidden_size_per_attention_head**-0.5

    def __call__(self, x, mask=None, cache=None):
        qkv = self.c_attn(x)

        q, k, v = mx.split(qkv, 3, axis=-1)

        B, L, _ = q.shape

        q = q.reshape(B, L, self.num_attention_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_attention_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_attention_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            k_cache, v_cache = cache
            q = self.rotary_emb(q, offset=k_cache.shape[2])
            k = self.rotary_emb(k, offset=k_cache.shape[2])
            k = mx.concatenate([k_cache, k], axis=2)
            v = mx.concatenate([v_cache, v], axis=2)

        else:
            q = self.rotary_emb(q)
            k = self.rotary_emb(k)

        q = q.astype(mx.float32)
        k = k.astype(mx.float32)

        scores = (q * self.scale) @ k.transpose(0, 1, 3, 2)

        if mask is not None:
            scores = scores + mask

        scores = mx.softmax(scores, axis=-1).astype(v.dtype)
        v_hat = (scores @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.c_proj(v_hat), (k, v)


class QWenMlp(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.w1 = nn.Linear(
            args.hidden_size, args.intermediate_size // 2, bias=not args.no_bias
        )
        self.w2 = nn.Linear(
            args.intermediate_size // 2, args.hidden_size, bias=not args.no_bias
        )
        self.c_proj = nn.Linear(
            args.intermediate_size // 2, args.hidden_size, bias=not args.no_bias
        )

    def __call__(self, x):
        a1 = self.w1(x)
        a2 = self.w2(x)
        intermediate_parallel = a1 * nn.silu(a2)
        out = self.c_proj(intermediate_parallel)
        return out


class QWenBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.ln_1 = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.attn = QWenAttntion(args)
        self.ln_2 = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.mlp = QWenMlp(args)

    def __call__(self, x, mask=None, cache=None):
        residual = x
        x = self.ln_1(x)
        x, cache = self.attn(x, mask=mask, cache=cache)
        residual = x + residual
        x = self.ln_2(residual)
        x = self.mlp(x)
        x = x + residual

        return x, cache


class QWen(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.embed_dim = args.hidden_size

        self.wte = nn.Embedding(args.vocab_size, args.hidden_size)
        self.h = [QWenBlock(args) for _ in range(args.num_hidden_layers)]
        self.ln_f = nn.RMSNorm(self.embed_dim, eps=args.layer_norm_epsilon)

        self.lm_head = nn.Linear(self.embed_dim, args.vocab_size, bias=False)

    def __call__(self, inputs, mask=None, cache=None):
        x = self.wte(inputs)

        mask = None
        if x.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(
                x.shape[1])
            mask = mask.astype(x.dtype)

        if cache is None:
            cache = [None] * len(self.h)

        for e, layer in enumerate(self.h):
            x, cache[e] = layer(x, mask, cache[e])

        x = self.ln_f(x)

        return self.lm_head(x), cache


def generate(prompt: mx.array, model: QWen, temp: 0.0):
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


def load_model(tokenizer_path: str = "Qwen/Qwen-1_8B"):
    model = QWen(ModelArgs())
    weights = mx.load("weights.npz")
    model.update(tree_unflatten(list(weights.items())))
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True)
    return model, tokenizer


if __name__ == "__main__":
    # The infernece code and arguments were mainly derived from phi-2 example.

    parser = argparse.ArgumentParser(description="Qwen inference script")
    parser.add_argument(
        "--tokenizer",
        help="The tokenizer to be used, defaults to Qwen/Qwen-1_8B",
        default="Qwen/Qwen-1_8B",
    )
    parser.add_argument(
        "--prompt",
        help="The message to be processed by the model",
        # The example from the official huggingface repo of Qwen
        default="蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n埃塞俄比亚的首都是",
    )
    parser.add_argument(
        "--max_tokens",
        "-m",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp",
        help="The sampling temperature.",
        type=float,
        default=0.0,
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")
    args = parser.parse_args()

    mx.random.seed(args.seed)

    model, tokenizer = load_model(args.tokenizer)

    prompt = tokenizer(
        args.prompt,
        return_tensors="np",
        return_attention_mask=False,
    )["input_ids"]

    prompt = mx.array(prompt)

    print(args.prompt, end="", flush=True)

    tokens = []
    for token, _ in zip(generate(prompt, model, args.temp), range(args.max_tokens)):
        tokens.append(token)

        if (len(tokens) % 10) == 0:
            mx.eval(tokens)
            eos_index = next(
                (i for i, t in enumerate(tokens)
                 if t.item() == tokenizer.eos_token_id),
                None,
            )

            if eos_index is not None:
                tokens = tokens[:eos_index]

            s = tokenizer.decode([t.item() for t in tokens])
            print(s, end="", flush=True)
            tokens = []
            if eos_index is not None:
                break

    mx.eval(tokens)
    s = tokenizer.decode([t.item() for t in tokens])
    print(s, flush=True)
