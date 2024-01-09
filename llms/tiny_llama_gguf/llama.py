# Copyright © 2023 Apple Inc.

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


import numpy as np
import mlx.core as mx
import mlx.nn as nn
from gguf.gguf_reader import GGUFReader
from mlx.utils import tree_flatten, tree_unflatten
from tokenizers import Tokenizer
from tokenizers.models import BPE


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


def get_field(gguf_reader: GGUFReader, key: str):
    f = gguf_reader.get_field(key)
    if f is None:
        return 0
    if len(f.data) != 1:
        raise NotImplementedError(f"multiple data is not supported")
    part = f.parts[f.data[0]]
    if len(part) != 1:
        raise NotImplementedError(f"multiple parts are not supported")
    value = part[0]
    if isinstance(value, np.float32):
        return float(value)
    elif isinstance(value, np.uint32):
        return int(value)
    return value


def get_string_array_field(gguf_reader: GGUFReader, key: str):
    f = gguf_reader.get_field(key)
    return [bytes(f.parts[d]).decode("utf-8") for d in f.data]


def get_config(gguf_reader: GGUFReader, weights: dict[str, mx.array]):
    output = {}
    output["dim"] = get_field(gguf_reader, "llama.embedding_length")
    output["n_layers"] = get_field(gguf_reader, "llama.block_count")
    output["n_heads"] = get_field(gguf_reader, "llama.attention.head_count")
    output["head_dim"] = output["dim"] // output["n_heads"]
    output["hidden_dim"] = get_field(gguf_reader, "llama.feed_forward_length")
    output["n_kv_heads"] = get_field(gguf_reader, "llama.attention.head_count_kv")
    output["norm_eps"] = get_field(
        gguf_reader, "llama.attention.layer_norm_rms_epsilon"
    )
    output["vocab_size"] = weights["output.weight"].shape[0]
    output["rope_theta"] = get_field(gguf_reader, "llama.rope.freq_base")
    output["rope_traditional"] = True
    return output


class GGUFTokenizer:
    @staticmethod
    def parse_token(token):
        if len(token) == 6 and token.startswith("<0x") and token.endswith(">"):
            return chr(int(token[3:5], 16))
        return token

    def __init__(self, gguf_reader):
        # TODO: do we need scores and token type?
        # tokenizer.ggml.scores: [array] [0.000000, 0.000000, 0.000000, ..
        # tokenizer.ggml.token_type: [array] [2, 3, 3, 6, ...
        tokens = get_string_array_field(gguf_reader, "tokenizer.ggml.tokens")
        vocab = {GGUFTokenizer.parse_token(t): i for i, t in enumerate(tokens)}
        merges = get_string_array_field(gguf_reader, "tokenizer.ggml.merges")
        merges = [tuple(m.split(" ")) for m in merges]
        model = BPE(vocab, merges, byte_fallback=True)
        self._tokenizer = Tokenizer(model)
        self._bos_token_id = get_field(gguf_reader, "bos_token_id")
        self._eos_token_id = get_field(gguf_reader, "eos_token_id")
        self._unknown_token_id = get_field(gguf_reader, "unknown_token_id")
        self._padding_token_id = get_field(gguf_reader, "padding_token_id")

    def bos_id(self):
        return self._bos_token_id

    def encode(self, input):
        return self._tokenizer.encode("▁" + input.replace(" ", "▁")).ids

    def decode(self, input):
        return self._tokenizer.decode(input).replace(" ", "").replace("▁", " ")


def translate_weight_names(name):
    name = name.replace("blk.", "layers.")
    name = name.replace("ffn_gate", "feed_forward.w1")
    name = name.replace("ffn_down", "feed_forward.w2")
    name = name.replace("ffn_up", "feed_forward.w3")
    name = name.replace("attn_q", "attention.wq")
    name = name.replace("attn_k", "attention.wk")
    name = name.replace("attn_v", "attention.wv")
    name = name.replace("attn_norm", "attention_norm")
    name = name.replace("attn_output", "attention.wo")
    name = name.replace("token_embd", "tok_embeddings")
    name = name.replace("output_norm", "norm")
    return name


def validate_weights(weights, model):
    current_weights = tree_flatten(model.parameters())
    weights_to_load = list(weights.items())
    current_weights_dict = dict(current_weights)
    current_weights_keys = set(current_weights_dict.keys())
    weights_to_load_dict = dict(weights_to_load)
    weights_to_load_keys = set(weights_to_load_dict.keys())
    missing = current_weights_keys - weights_to_load_keys
    if missing:
        print("Missing weights: ", sorted(missing))
    ignored = weights_to_load_keys - current_weights_keys
    if ignored:
        print("Weights ignored: ", sorted(ignored))
    shared = current_weights_keys & weights_to_load_keys
    for key in sorted(shared):
        if weights_to_load_dict[key].shape != current_weights_dict[key].shape:
            print("Shape mismatch for key: ", key)
            print("Expected shape: ", current_weights_dict[key].shape)
            print("Loading shape: ", weights_to_load_dict[key].shape)


def load_model(gguf_path):
    print("[INFO] Loading model from {}.".format(gguf_path))
    weights = mx.load(str(gguf_path))
    weights = {translate_weight_names(k): v for k, v in weights.items()}

    reader = GGUFReader(str(gguf_path))
    config = get_config(reader, weights)
    model = Llama(ModelArgs(**config))
    validate_weights(weights, model)
    model.update(tree_unflatten(list(weights.items())))

    tokenizer = GGUFTokenizer(reader)
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama inference script")
    parser.add_argument(
        "--model-path",
        help="Path to the gguf file",
    )
    parser.add_argument(
        "--prompt",
        help="The message to be processed by the model. Ignored when --few-shot is provided.",
        default="""<|system|>
You are a helpful assistant</s>
<|user|>
Can you describe the taste of an Apple?</s>
<|assistant|>""",
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
    generate(args)
