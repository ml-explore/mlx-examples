# Copyright © 2023 Apple Inc.

import argparse
import glob
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from sentencepiece import SentencePieceProcessor


@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int
    rope_theta: float
    rope_traditional: bool = True


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
        self.args = args

        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim**-0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)
        self.rope = nn.RoPE(
            args.head_dim, traditional=args.rope_traditional, base=args.rope_theta
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        B, L, D = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        def repeat(a):
            a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)
            return a.reshape([B, self.n_heads, L, -1])

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

        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

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


class Llama(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def __call__(self, x):
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.tok_embeddings.weight.dtype)

        x = self.tok_embeddings(x)
        for l in self.layers:
            x, _ = l(x, mask)
        x = self.norm(x)
        return self.output(x)

    def generate(self, x, temp=1.0):
        def sample(logits):
            if temp == 0:
                return mx.argmax(logits, axis=-1)
            else:
                return mx.random.categorical(logits * (1 / temp))

        cache = []

        # Make an additive causal mask. We will need that to process the prompt.
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.tok_embeddings.weight.dtype)

        # First we process the prompt x the same was as in __call__ but
        # save the caches in cache
        x = self.tok_embeddings(x)
        for l in self.layers:
            x, c = l(x, mask=mask)
            # We store the per layer cache in a simple python list
            cache.append(c)
        x = self.norm(x)
        # We only care about the last logits that generate the next token
        y = self.output(x[:, -1])
        y = sample(y)

        # y now has size [1]
        # Since MLX is lazily evaluated nothing is computed yet.
        # Calling y.item() would force the computation to happen at
        # this point but we can also choose not to do that and let the
        # user choose when to start the computation.
        yield y

        # Now we parsed the prompt and generated the first token we
        # need to feed it back into the model and loop to generate the
        # rest.
        while True:
            # Unsqueezing the last dimension to add a sequence length
            # dimension of 1
            x = y[:, None]

            x = self.tok_embeddings(x)
            for i in range(len(cache)):
                # We are overwriting the arrays in the cache list. When
                # the computation will happen, MLX will be discarding the
                # old cache the moment it is not needed anymore.
                x, cache[i] = self.layers[i](x, mask=None, cache=cache[i])
            x = self.norm(x)
            y = sample(self.output(x[:, -1]))

            yield y


def tic():
    return time.time()


def toc(msg, start):
    end = time.time()
    return f"[INFO] {msg}: {end - start:.3f} s"


def generate(args):
    input("Press enter to start generation")
    print("------")
    print(args.prompt)
    x = mx.array([[tokenizer.bos_id()] + tokenizer.encode(args.prompt)])
    skip = 0
    prompt_processing = None
    tokens = []
    start = tic()
    for token in model.generate(x, args.temp):
        tokens.append(token)

        if len(tokens) == 1:
            # Actually perform the computation to measure the prompt processing time
            mx.eval(token)
            prompt_processing = toc("Prompt processing", start)

        if len(tokens) >= args.max_tokens:
            break

        elif (len(tokens) % args.write_every) == 0:
            # It is perfectly ok to eval things we have already eval-ed.
            mx.eval(tokens)
            s = tokenizer.decode([t.item() for t in tokens])
            print(s[skip:], end="", flush=True)
            skip = len(s)

    mx.eval(tokens)
    full_gen = toc("Full generation", start)
    s = tokenizer.decode([t.item() for t in tokens])
    print(s[skip:], flush=True)
    print("------")
    print(prompt_processing)
    print(full_gen)


def few_shot_generate(args):
    def possible_end(s):
        word = "[Instruction]"
        for i in range(len(word) - 1, 0, -1):
            if s[-i:] == word[:i]:
                return 0
        if s[-len(word) :] == word:
            return 1
        return -1

    def generate(question):
        x = mx.array([[tokenizer.bos_id()] + tokenizer.encode(question)])
        skip = 0
        prompt_processing = None
        tokens = []
        start = tic()
        for token in model.generate(x, args.temp):
            tokens.append(token)

            if len(tokens) == 1:
                # Actually perform the computation to measure the prompt processing time
                mx.eval(token)
                prompt_processing = toc("Prompt processing", start)

            if len(tokens) >= args.max_tokens:
                break

            mx.eval(tokens)
            token_list = [t.item() for t in tokens]
            s = tokenizer.decode(token_list)

            end = possible_end(s)
            if end == 0:
                continue
            if end == 1:
                skip = len(s)
                break

            print(s[skip:], end="", flush=True)
            skip = len(s)
            if token_list[-1] == tokenizer.eos_id():
                break

        mx.eval(tokens)
        full_gen = toc("Full generation", start)
        s = tokenizer.decode([t.item() for t in tokens])
        print(s[skip:], end="", flush=True)

    print("[INFO] Loading few-shot examples from: {}".format(args.few_shot))
    prompt = open(args.few_shot).read().strip()
    while True:
        question = input("Ask a question: ")
        generate(prompt.replace("{}", question))
        print()


def sanitize_config(config, weights):
    config.pop("model_type", None)
    n_heads = config["n_heads"]
    if "n_kv_heads" not in config:
        config["n_kv_heads"] = n_heads
    if "head_dim" not in config:
        config["head_dim"] = config["dim"] // n_heads
    if "hidden_dim" not in config:
        config["hidden_dim"] = weights["layers.0.feed_forward.w1.weight"].shape[0]
    if config.get("vocab_size", -1) < 0:
        config["vocab_size"] = weights["output.weight"].shape[-1]
    if "rope_theta" not in config:
        config["rope_theta"] = 10000
    unused = ["multiple_of", "ffn_dim_multiplier"]
    for k in unused:
        config.pop(k, None)
    return config


def load_model(model_path):
    model_path = Path(model_path)

    unsharded_weights_path = Path(model_path / "weights.npz")
    if unsharded_weights_path.is_file():
        print("[INFO] Loading model from {}.".format(unsharded_weights_path))
        weights = mx.load(str(unsharded_weights_path))
    else:
        sharded_weights_glob = str(model_path / "weights.*.npz")
        weight_files = glob.glob(sharded_weights_glob)
        print("[INFO] Loading model from {}.".format(sharded_weights_glob))

        if len(weight_files) == 0:
            raise FileNotFoundError("No weights found in {}".format(model_path))

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf).items())

    with open(model_path / "config.json", "r") as f:
        config = sanitize_config(json.loads(f.read()), weights)
        quantization = config.pop("quantization", None)
    model = Llama(ModelArgs(**config))
    if quantization is not None:
        nn.QuantizedLinear.quantize_module(model, **quantization)
    model.update(tree_unflatten(list(weights.items())))
    tokenizer = SentencePieceProcessor(model_file=str(model_path / "tokenizer.model"))
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama inference script")
    parser.add_argument(
        "--model-path",
        help="Path to the model weights and tokenizer",
        default="mlx_model",
    )
    parser.add_argument(
        "--prompt",
        help="The message to be processed by the model. Ignored when --few-shot is provided.",
        default="In the beginning the Universe was created.",
    )
    parser.add_argument(
        "--few-shot",
        help="Read a few shot prompt from a file (as in `sample_prompt.txt`).",
    )
    parser.add_argument(
        "--max-tokens", "-m", type=int, default=100, help="How many tokens to generate"
    )
    parser.add_argument(
        "--write-every", type=int, default=1, help="After how many tokens to detokenize"
    )
    parser.add_argument(
        "--temp", type=float, default=0.0, help="The sampling temperature"
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")

    args = parser.parse_args()

    mx.random.seed(args.seed)

    model, tokenizer = load_model(args.model_path)
    if args.few_shot:
        few_shot_generate(args)
    else:
        generate(args)
