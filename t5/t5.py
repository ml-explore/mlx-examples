import argparse
import glob
import math
from pathlib import Path
from time import perf_counter_ns
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from mlx.utils import tree_map, tree_unflatten
from transformers import AutoTokenizer, T5Config


def _relative_position_bucket(
    relative_position, bidirectional=True, num_buckets=32, max_distance=128
):
    """
    Adapted from HF Tensorflow:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py

    Translate relative position to a bucket number for relative attention. The relative position is defined as
    memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
    position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
    small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
    positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
    This should allow for more graceful generalization to longer sequences than the model has been trained on

    Args:
        relative_position: an int32 Tensor
        bidirectional: a boolean - whether the attention is bidirectional
        num_buckets: an integer
        max_distance: an integer

    Returns:
        a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
    """
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).astype(mx.int16) * num_buckets
        relative_position = mx.abs(relative_position)
    else:
        relative_position = -mx.minimum(
            relative_position, mx.zeros_like(relative_position)
        )
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    scale = (num_buckets - max_exact) / np.log(max_distance / max_exact)
    relative_position_if_large = max_exact + (
        mx.log(relative_position.astype(mx.float32) / max_exact) * scale
    ).astype(mx.int16)
    relative_position_if_large = mx.minimum(relative_position_if_large, num_buckets - 1)
    relative_buckets += mx.where(
        is_small, relative_position, relative_position_if_large
    )
    return relative_buckets


class RelativePositionBias(nn.Module):
    def __init__(self, config: T5Config, bidirectional: bool):
        self.bidirectional = bidirectional
        self.num_buckets = config.relative_attention_num_buckets
        self.max_distance = config.relative_attention_max_distance
        self.n_heads = config.num_heads

    def __call__(self, embeddings: nn.Embedding, query_length: int, key_length: int, offset: int = 0):
        """Compute binned relative position bias"""
        context_position = mx.arange(offset, query_length)[:, None]
        memory_position = mx.arange(key_length)[None, :]

        # shape (query_length, key_length)
        relative_position = memory_position - context_position
        relative_position_bucket = _relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )

        # shape (query_length, key_length, num_heads)
        values = embeddings(relative_position_bucket)

        # shape (num_heads, query_length, key_length)
        return values.transpose(2, 0, 1)


class MultiHeadAttention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool):
        super().__init__()
        inner_dim = config.d_kv * config.num_heads
        self.num_heads = config.num_heads
        self.q = nn.Linear(config.d_model, inner_dim, bias=False)
        self.k = nn.Linear(config.d_model, inner_dim, bias=False)
        self.v = nn.Linear(config.d_model, inner_dim, bias=False)
        self.o = nn.Linear(inner_dim, config.d_model, bias=False)
        if has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(config.relative_attention_num_buckets, self.num_heads)

    def __call__(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> [mx.array, Tuple[mx.array, mx.array]]:
        queries = self.q(queries)
        keys = self.k(keys)
        values = self.v(values)

        num_heads = self.num_heads
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            key_cache, value_cache = cache
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)

        # scale = math.sqrt(1 / queries.shape[-1])
        # output = mx.fast.scaled_dot_product_attention(
        #     queries, keys, values, scale=scale, mask=mask
        # )
        # output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        scores = queries @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores += mask.astype(scores.dtype)
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o(output), (keys, values)


class LayerSelfAttention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool):
        super().__init__()
        self.SelfAttention = MultiHeadAttention(config, has_relative_attention_bias)
        self.layer_norm = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> [mx.array, Tuple[mx.array, mx.array]]:
        y = self.layer_norm(x)
        return self.SelfAttention(y, y, y, mask, cache)


class LayerCrossAttention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool):
        super().__init__()
        self.EncDecAttention = MultiHeadAttention(config, has_relative_attention_bias)
        self.layer_norm = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)

    def __call__(
        self,
        x: mx.array,
        memory: mx.array,
        memory_mask: Optional[mx.array],
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> [mx.array, Tuple[mx.array, mx.array]]:
        y = self.layer_norm(x)
        return self.EncDecAttention(y, memory, memory, memory_mask, cache)


class DenseActivation(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        mlp_dims = config.d_ff or config.d_model * 4
        self.gated = config.feed_forward_proj.startswith("gated")
        if self.gated:
            self.wi_0 = nn.Linear(config.d_model, mlp_dims, bias=False)
            self.wi_1 = nn.Linear(config.d_model, mlp_dims, bias=False)
        else:
            self.wi = nn.Linear(config.d_model, mlp_dims, bias=False)
        self.wo = nn.Linear(mlp_dims, config.d_model, bias=False)
        activation = config.feed_forward_proj.removeprefix("gated-")
        if activation == "relu":
            self.act = nn.relu
        elif activation == "gelu":
            self.act = nn.gelu
        elif activation == "silu":
            self.act = nn.silu
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def __call__(self, x):
        if self.gated:
            hidden_act = self.act(self.wi_0(x))
            hidden_linear = self.wi_1(x)
            x = hidden_act * hidden_linear
        else:
            x = self.act(self.wi(x))
        return self.wo(x)


class LayerFF(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.DenseReluDense = DenseActivation(config)
        self.layer_norm = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)

    def __call__(self, x: mx.array) -> [mx.array, Tuple[mx.array, mx.array]]:
        return self.DenseReluDense(self.layer_norm(x))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool):
        super().__init__()
        self.layer = [
            LayerSelfAttention(config, has_relative_attention_bias),
            LayerFF(config)
        ]

    def __call__(self, x, mask):
        y, _ = self.layer[0](x, mask=mask)
        x = x + y
        y = self.layer[1](x)
        x = x + y
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.block = [
            TransformerEncoderLayer(config, i == 0) for i in range(config.num_layers)
        ]
        self.final_layer_norm = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.relative_attention_bias = RelativePositionBias(config, bidirectional=True)

    def __call__(self, x: mx.array):
        pos_bias = self.relative_attention_bias(
            self.block[0].layer[0].SelfAttention.relative_attention_bias,
            x.shape[1],
            x.shape[1])
        for layer in self.block:
            x = layer(x, mask=pos_bias)
        return self.final_layer_norm(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool):
        super().__init__()
        self.layer = [
            LayerSelfAttention(config, has_relative_attention_bias),
            LayerCrossAttention(config, has_relative_attention_bias),
            LayerFF(config)
        ]

    def __call__(
        self,
        x: mx.array,
        memory: mx.array,
        mask: mx.array,
        memory_mask: mx.array,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ):
        y, cache = self.layer[0](x, mask, cache)
        x = x + y
        y, _ = self.layer[1](x, memory, memory_mask)
        x = x + y
        y = self.layer[2](x)
        x = x + y
        return x, cache


class TransformerDecoder(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        n_layers = getattr(config, "num_decoder_layers", config.num_layers)
        self.block = [
            TransformerDecoderLayer(config, i == 0) for i in range(n_layers)
        ]
        self.final_layer_norm = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.relative_attention_bias = RelativePositionBias(config, bidirectional=False)

    def __call__(self, x, memory, mask, memory_mask, cache=None):
        if cache is not None:
            offset = cache[0][0].shape[2]
        else:
            offset = 0
            cache = [None] * len(self.block)

        T = offset + x.shape[1]
        pos_bias = self.relative_attention_bias(
            self.block[0].layer[0].SelfAttention.relative_attention_bias,
            T,
            T,
            offset)
        if mask is not None:
            mask += pos_bias
        else:
            mask = pos_bias

        for e, layer in enumerate(self.block):
            x, cache[e] = layer(x, memory, mask, memory_mask, cache=cache[e])
        x = self.final_layer_norm(x)

        return x, cache


class T5(nn.Module):
    def __init__(self, config: T5Config):
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.tie_word_embeddings = config.tie_word_embeddings
        if not self.tie_word_embeddings:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.model_dim = config.d_model

    def encode(self, inputs: mx.array):
        return self.encoder(self.shared(inputs))

    def decode(
        self,
        inputs: mx.array,
        memory: mx.array,
        cache=None,
    ):
        inputs = self.shared(inputs)
        T = inputs.shape[1]
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            mask = mask.astype(inputs.dtype)
        else:
            mask = None

        y, cache = self.decoder(
            inputs, memory=memory, mask=mask, memory_mask=None, cache=cache
        )
        if not self.tie_word_embeddings:
            y = self.lm_head(y)
        else:
            y *= self.model_dim**-0.5
            y = y @ self.shared.weight.T
        return y, cache

    def __call__(
        self,
        inputs: mx.array,
        decoder_inputs: mx.array,
    ):
        return self.decode(decoder_inputs, self.encode(inputs))[0]


    @staticmethod
    def from_pretrained(config: T5Config, path: Path):
        weight_files = glob.glob(str(path / "*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors found in {path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        model = T5(config)
        model.load_weights(list(weights.items()))
        return model


class Tokenizer:
    def __init__(self, config: T5Config):
        self._decoder_start_id = config.decoder_start_token_id
        self._tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            legacy=False,
            model_max_length=getattr(config, "n_positions", 512),
        )

    @property
    def eos_id(self) -> int:
        return self._tokenizer.eos_token_id

    @property
    def decoder_start_id(self) -> int:
        return self._decoder_start_id

    def encode(self, s: str) -> mx.array:
        return mx.array(
            self._tokenizer(
                s,
                return_tensors="np",
                return_attention_mask=False,
            )["input_ids"]
        )

    def decode(self, t: List[int], with_sep: bool = True) -> str:
        tokens = self._tokenizer.convert_ids_to_tokens(t)
        return "".join(t.replace("▁", " " if with_sep else "") for t in tokens)


def generate(prompt: str, model: T5, tokenizer: Tokenizer, temp: Optional[float] = 0.0):
    def sample(logits):
        if temp == 0:
            return mx.argmax(logits, axis=-1)
        else:
            return mx.random.categorical(logits * (1 / temp))

    prompt = tokenizer.encode(prompt)
    decoder_inputs = mx.array([tokenizer.decoder_start_id])
    memory = model.encode(prompt)
    cache = None
    y = decoder_inputs
    while True:
        logits, cache = model.decode(y[None], memory, cache=cache)
        y = sample(logits[:, -1, :])
        yield y.squeeze()


def load_model(path_or_hf_repo: str, dtype: str = "float16"):
    path = Path(path_or_hf_repo)
    if not path.exists():
        path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                ],
            )
        )
    config = T5Config.from_pretrained(path)
    dtype = getattr(mx, dtype)
    model = T5.from_pretrained(config, path)
    mx.eval(model.parameters())
    return model, Tokenizer(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T5 Inference script")
    parser.add_argument(
        "--model",
        type=str,
        help="Name of the T5 model.",
        default="t5-small",
    )
    parser.add_argument(
        "--prompt",
        help="",
        default="translate English to German: That is good.",
    )
    parser.add_argument(
        "--encode-only",
        action="store_true",
        default=False,
        help="Whether to decode or not. If true, will output last layer of encoder.",
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
        default=0.0,
    )
    parser.add_argument(
        "--dtype",
        help="The model data type.",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )

    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")
    args = parser.parse_args()

    mx.random.seed(args.seed)

    model, tokenizer = load_model(args.model, args.dtype)

    if args.encode_only:
        print("[INFO] Encoding with T5...", flush=True)
        print(args.prompt, flush=True)
        encoder_output = model.encode(tokenizer.encode(args.prompt))
        print(encoder_output, flush=True)
        exit(0)

    print("[INFO] Generating with T5...", flush=True)
    print("Input: ", args.prompt, flush=True)

    start = perf_counter_ns()
    for token, n_tokens in zip(
        generate(args.prompt, model, tokenizer, args.temp), range(args.max_tokens)
    ):
        if token.item() == tokenizer.eos_id:
            break
        print(
            tokenizer.decode([token.item()], with_sep=n_tokens > 0),
            end="",
            flush=True,
        )

    n_tokens += 1
    end = perf_counter_ns()
    elapsed = (end - start) / 1.0e9
    print()
    print(f"Time: {elapsed:.2f} seconds, tokens/s: {n_tokens / elapsed:.2f}")
