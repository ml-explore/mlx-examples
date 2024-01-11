# Copyright © 2023 Apple Inc.

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int = None
    rope_theta: float = 10000
    rope_traditional: bool = False
    model_type: str = None
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_scaling:
            required_keys = {"factor", "type"}
            if not all(key in self.rope_scaling for key in required_keys):
                raise ValueError(f"rope_scaling must contain keys {required_keys}")

            if self.rope_scaling["type"] != "linear":
                raise ValueError("rope_scaling 'type' currently only supports 'linear'")

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


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

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.repeats = n_heads // n_kv_heads

        head_dim = args.hidden_size // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        rope_scale = (
            1 / args.rope_scaling["factor"]
            if args.rope_scaling is not None and args.rope_scaling["type"] == "linear"
            else 1
        )
        self.rope = nn.RoPE(
            head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
            scale=rope_scale,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        def repeat(a):
            a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)
            return a.reshape([B, self.n_heads, L, -1])

        if self.repeats > 1:
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
        return self.o_proj(output), (keys, values)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, cache


class LlamaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, mask, cache[e])

        return self.norm(h), cache


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model = LlamaModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out, cache = self.model(inputs, cache)
        return self.lm_head(out), cache


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
    if f is None:
        return []
    return [bytes(f.parts[d]).decode("utf-8") for d in f.data]


def get_config(gguf_reader: GGUFReader, weights: dict[str, mx.array]):
    output = {}
    output["hidden_size"] = get_field(gguf_reader, "llama.embedding_length")
    output["num_hidden_layers"] = get_field(gguf_reader, "llama.block_count")
    output["num_attention_heads"] = get_field(gguf_reader, "llama.attention.head_count")
    output["intermediate_size"] = get_field(gguf_reader, "llama.feed_forward_length")
    output["num_key_value_heads"] = get_field(
        gguf_reader, "llama.attention.head_count_kv"
    )
    output["rms_norm_eps"] = get_field(
        gguf_reader, "llama.attention.layer_norm_rms_epsilon"
    )
    output["vocab_size"] = weights["lm_head.weight"].shape[0]
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
        scores = gguf_reader.get_field("tokenizer.ggml.scores")
        scores = [scores.parts[d].item() for d in scores.data]

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
    name = name.replace("blk.", "model.layers.")
    name = name.replace("ffn_gate", "mlp.gate_proj")
    name = name.replace("ffn_down", "mlp.down_proj")
    name = name.replace("ffn_up", "mlp.up_proj")
    name = name.replace("attn_q", "self_attn.q_proj")
    name = name.replace("attn_k", "self_attn.k_proj")
    name = name.replace("attn_v", "self_attn.v_proj")
    name = name.replace("attn_output", "self_attn.o_proj")
    name = name.replace("attn_norm", "input_layernorm")
    name = name.replace("ffn_norm", "post_attention_layernorm")
    name = name.replace("token_embd", "model.embed_tokens")
    name = name.replace("output_norm", "model.norm")
    name = name.replace("output", "lm_head")
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
    reader = GGUFReader(gguf_file)
    tokenizer = GGUFTokenizer(reader)
    weights = mx.load(gguf_file)
    weights = {translate_weight_names(k): v for k, v in weights.items()}
    config = get_config(reader, weights)
    model = Model(ModelArgs(**config))
    model.load_weights(list(weights.items()))
    return model, tokenizer


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
