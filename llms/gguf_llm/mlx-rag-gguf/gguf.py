# Copyright © 2023 Apple Inc.
# Edited by: Jaward Sesay (Jaykef) 2024-26-04
# File: gguf.py - loads and inferences .gguf models

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import utils
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten, tree_unflatten

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

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
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
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
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
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

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


def get_config(metadata: dict):
    output = {
        "hidden_size": metadata["llama.embedding_length"],
        "num_hidden_layers": metadata["llama.block_count"],
        "num_attention_heads": metadata["llama.attention.head_count"],
        "intermediate_size": metadata["llama.feed_forward_length"],
        "num_key_value_heads": metadata["llama.attention.head_count_kv"],
        "rms_norm_eps": metadata["llama.attention.layer_norm_rms_epsilon"],
        "vocab_size": len(metadata["tokenizer.ggml.tokens"]),
        "rope_theta": metadata["llama.rope.freq_base"],
        "rope_traditional": True,
    }
    output = {k: v.item() if isinstance(v, mx.array) else v for k, v in output.items()}
    return output


class GGUFTokenizer:
    def __init__(self, metadata):
        self._tokenizer = utils.spm_tokenizer(metadata)

    def encode(self, s: str) -> mx.array:
        return mx.array([self._tokenizer.bos_id()] + self._tokenizer.encode(s))

    @property
    def eos_token_id(self):
        return self._tokenizer.eos_id()

    def decode(self, toks: List[int]) -> str:
        return self._tokenizer.decode(toks)


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
        if not (Path(model_path) / gguf_file).exists():
            raise ValueError(f"File {gguf_file} not in repo {repo}.")
        gguf_file = str(Path(model_path) / gguf_file)

    print(f"[INFO] Loading model from {gguf_file}")
    weights, metadata = mx.load(gguf_file, return_metadata=True)
    gguf_ft = metadata["general.file_type"]
    if gguf_ft == 0 or gguf_ft == 1:
        # ALL_F32 or MOSTLY_F16
        quantization = None
        pass
    elif gguf_ft == 2 or gguf_ft == 3:
        # MOSTLY_Q4_0 or MOSTLY_Q4_1
        quantization = {"group_size": 32, "bits": 4}
    elif gguf_ft == 7:
        # MOSTLY_Q8_0 = 7
        quantization = {"group_size": 32, "bits": 8}
    else:
        quantization = None
        print("[WARNING] Using unsupported GGUF quantization. Casting to float16.")

    weights = {translate_weight_names(k): v for k, v in weights.items()}
    config = get_config(metadata)
    model = Model(ModelArgs(**config))
    if quantization is not None:
        # quantized the LM head?
        qm = model if "lm_head.scales" in weights else model.model
        nn.QuantizedLinear.quantize_module(
            qm,
            **quantization,
        )

    def dequantize(k):
        weight = weights.pop(f"{k}.weight")
        scales = weights.pop(f"{k}.scales")
        biases = weights.pop(f"{k}.biases")
        weights[f"{k}.weight"] = mx.dequantize(
            weight, scales=scales, biases=biases, **quantization
        )

    # Dequantize embeddings
    dequantize("model.embed_tokens")

    tokenizer = GGUFTokenizer(metadata)
    model.load_weights(list(weights.items()))
    return model, tokenizer


def generate(prompt: mx.array, model: Model, temp: float = 0.0, max_tokens: int = 512, verbose: bool = False):
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