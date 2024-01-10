# Copyright © 2023 Apple Inc.

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from gguf.gguf_reader import GGUFReader
from huggingface_hub import snapshot_download
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


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.tok_embeddings(inputs)

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, mask, cache[e])

        return self.output(self.norm(h)), cache


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
    def __init__(self, gguf_reader):
        def parse_token(token):
            if len(token) == 6 and token.startswith("<0x") and token.endswith(">"):
                return chr(int(token[3:5], 16))
            return token

        # TODO: do we need scores and token type?
        # tokenizer.ggml.scores: [array] [0.000000, 0.000000, 0.000000, ..
        # tokenizer.ggml.token_type: [array] [2, 3, 3, 6, ...
        tokens = get_string_array_field(gguf_reader, "tokenizer.ggml.tokens")
        vocab = {parse_token(t): i for i, t in enumerate(tokens)}
        merges = get_string_array_field(gguf_reader, "tokenizer.ggml.merges")
        merges = [tuple(m.split(" ")) for m in merges]
        model = BPE(vocab, merges, byte_fallback=True)
        self._tokenizer = Tokenizer(model)
        self._bos_token_id = get_field(gguf_reader, "bos_token_id")
        self._eos_token_id = get_field(gguf_reader, "eos_token_id")

    def encode(self, s: str) -> mx.array:
        return mx.array(
            [self._bos_token_id] + self._tokenizer.encode("▁" + s.replace(" ", "▁")).ids
        )

    @property
    def eos_token_id(self):
        return self._eos_token_id

    def decode(self, toks: List[int]) -> str:
        return self._tokenizer.decode(toks).replace(" ", "").replace("▁", " ")


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


def load(gguf_file: str, repo: str = None):
    # If the gguf_file exists, try to load model from it.
    # Otherwise try to download and cache from the HF repo
    if not Path(gguf_file).exists():
        if repo is None:
            raise ValueError(
                f"Could not find file {gguf_file}, and no Hugging Face"
                " repo provided for download."
            )
        model_path = snapshot_download(
            repo_id=repo,
            allow_patterns=[gguf_file],
        )
        gguf_file = str(Path(model_path) / gguf_file)

    print(f"[INFO] Loading model from {gguf_file}")
    weights = mx.load(gguf_file)
    import pdb

    pdb.set_trace()
    weights = {translate_weight_names(k): v for k, v in weights.items()}

    reader = GGUFReader(gguf_file)
    config = get_config(reader, weights)
    model = Model(ModelArgs(**config))
    model.load_weights(list(weights.items()))
    return model, GGUFTokenizer(reader)


def generate(prompt: mx.array, model: Model, temp: float = 0.0):
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
