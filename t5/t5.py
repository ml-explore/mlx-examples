import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List
from typing import Optional
from time import perf_counter_ns

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from transformers import AutoTokenizer


@dataclass
class ModelArgs:
    d_ff: int = 2048
    d_kv: int = 64
    d_model: int = 512
    dropout_rate: int = 0.1
    layer_norm_epsilon: float = 1e-06
    n_positions: int = 512
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    num_heads: int = 8
    num_layers: int = 6
    decoder_start_token_id: int = 0
    eos_token_id: int = 1
    pad_token_id: int = 0
    vocab_size: int = 32128


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
    scale = np.log(max_distance / max_exact) * (num_buckets - max_exact)
    relative_position_if_large = max_exact + (
        mx.log(relative_position.astype(mx.float32) / max_exact) * scale
    ).astype(mx.int16)
    relative_position_if_large = mx.minimum(relative_position_if_large, num_buckets - 1)
    relative_buckets += mx.where(
        is_small, relative_position, relative_position_if_large
    )
    return relative_buckets


class RelativePositionBias(nn.Module):
    def __init__(self, config: ModelArgs, bidirectional: bool):
        self.bidirectional = False  # bidirectional
        self.num_buckets = config.relative_attention_num_buckets
        self.max_distance = config.relative_attention_max_distance
        self.n_heads = config.num_heads
        self.embeddings = nn.Embedding(
            config.relative_attention_num_buckets, config.num_heads
        )

    def __call__(self, query_length: int, key_length: int, offset: int = 0):
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
        values = self.embeddings(relative_position_bucket)

        # shape (num_heads, query_length, key_length)
        return values.transpose(2, 0, 1)


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.num_heads = config.num_heads
        self.query_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.key_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.value_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def __call__(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        mask: mx.array,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> [mx.array, Tuple[mx.array, mx.array]]:
        queries = self.query_proj(queries)
        keys = self.key_proj(keys)
        values = self.value_proj(values)

        num_heads = self.num_heads
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, num_heads, -1).transpose(0, 2, 3, 1)
        values = values.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            key_cache, value_cache = cache
            keys = mx.concatenate([key_cache, keys], axis=3)
            values = mx.concatenate([value_cache, values], axis=2)

        # Dimensions are [batch x num heads x sequence x hidden dim]
        scores = queries @ keys
        if mask is not None:
            scores = scores + mask.astype(scores.dtype)

        scores = mx.softmax(scores, axis=-1)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(values_hat), (keys, values)


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


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        mlp_dims = config.d_ff or config.d_model * 4
        self.attention = MultiHeadAttention(config)
        self.ln1 = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.ln2 = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.linear1 = nn.Linear(config.d_model, mlp_dims, bias=False)
        self.linear2 = nn.Linear(mlp_dims, config.d_model, bias=False)

    def __call__(self, x, mask):
        y = self.ln1(x)
        y, _ = self.attention(y, y, y, mask=mask)
        x = x + y

        y = self.ln2(x)
        y = self.linear1(y)
        y = mx.maximum(y, 0)
        y = self.linear2(y)
        return x + y


class TransformerEncoder(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.layers = [
            TransformerEncoderLayer(config) for i in range(config.num_layers)
        ]
        self.ln = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.relative_attention_bias = RelativePositionBias(config, bidirectional=True)

    def __call__(self, x: mx.array):
        pos_bias = self.relative_attention_bias(x.shape[1], x.shape[1])
        for layer in self.layers:
            x = layer(x, mask=pos_bias)
        return self.ln(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        mlp_dims = config.d_ff or config.d_model * 4
        self.self_attention = MultiHeadAttention(config)
        self.cross_attention = MultiHeadAttention(config)
        self.ln1 = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.ln2 = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.ln3 = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.linear1 = nn.Linear(config.d_model, mlp_dims, bias=False)
        self.linear2 = nn.Linear(mlp_dims, config.d_model, bias=False)

    def __call__(
        self,
        x: mx.array,
        memory: mx.array,
        mask: mx.array,
        memory_mask: mx.array,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ):
        y = self.ln1(x)
        y, cache = self.self_attention(y, y, y, mask, cache)
        x = x + y

        y = self.ln2(x)
        y, _ = self.cross_attention(x, memory, memory, memory_mask)
        x = x + y

        y = self.ln3(x)
        y = self.linear1(y)
        y = mx.maximum(y, 0)
        y = self.linear2(y)
        x = x + y

        return x, cache


class TransformerDecoder(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.layers = [
            TransformerDecoderLayer(config) for i in range(config.num_layers)
        ]
        self.ln = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.relative_attention_bias = RelativePositionBias(config, bidirectional=False)

    def __call__(self, x, memory, mask, memory_mask, cache=None):
        if cache is not None:
            offset = cache[0][0].shape[3]
        else:
            offset = 0
            cache = [None] * len(self.layers)

        T = offset + x.shape[1]
        pos_bias = self.relative_attention_bias(T, T, offset=offset)
        if mask is not None:
            mask += pos_bias
        else:
            mask = pos_bias

        for e, layer in enumerate(self.layers):
            x, cache[e] = layer(x, memory, mask, memory_mask, cache=cache[e])
        x = self.ln(x)

        return x, cache


class OutputHead(nn.Module):
    def __init__(self, config: ModelArgs):
        self.linear = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def __call__(self, inputs):
        return self.linear(inputs)


class T5(nn.Module):
    def __init__(self, config: ModelArgs):
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.lm_head = OutputHead(config)

    def encode(self, inputs: mx.array):
        return self.encoder(self.wte(inputs))

    def decode(
        self,
        inputs: mx.array,
        memory: mx.array,
        cache=None,
    ):
        inputs = self.wte(inputs)
        T = inputs.shape[1]
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            mask = mask.astype(inputs.dtype)
        else:
            mask = None

        y, cache = self.decoder(
            inputs, memory=memory, mask=mask, memory_mask=None, cache=cache
        )
        return self.lm_head(y), cache

    def __call__(
        self,
        inputs: mx.array,
        decoder_inputs: mx.array,
    ):
        return self.decode(decoder_inputs, self.encode(inputs))[0]


def generate(
    inputs: mx.array, decoder_inputs: mx.array, model: T5, temp: Optional[float] = 0.0
):
    def sample(logits):
        if temp == 0:
            return mx.argmax(logits, axis=-1)
        else:
            return mx.random.categorical(logits * (1 / temp))

    memory = model.encode(inputs)
    cache = None
    y = decoder_inputs
    while True:
        logits, cache = model.decode(y[None], memory, cache=cache)
        y = sample(logits[:, -1, :])
        yield y.squeeze()


def load_model(model_config):
    model = T5(model_config)
    weights = mx.load("weights.npz")
    current_weights = tree_flatten(model.parameters())
    weights_to_load = list(weights.items())
    current_weights_dict = dict(current_weights)
    current_weights_keys = set(current_weights_dict.keys())
    weights_to_load_dict = dict(weights_to_load)
    weights_to_load_keys = set(weights_to_load_dict.keys())
    print("Missing weights: ", sorted(current_weights_keys - weights_to_load_keys))
    print()
    print("Weights ignored: ", sorted(weights_to_load_keys - current_weights_keys))
    for key in current_weights_keys & weights_to_load_keys:
        if weights_to_load_dict[key].shape != current_weights_dict[key].shape:
            print("Shape mismatch for key: ", key)
            print("Expected shape: ", current_weights_dict[key].shape)
            print("Loading shape: ", weights_to_load_dict[key].shape)
    model.update(tree_unflatten(weights_to_load))
    mx.eval(model.parameters())
    tokenizer = AutoTokenizer.from_pretrained("t5-small", trust_remote_code=True)
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T5 Inference script")
    parser.add_argument(
        "--prompt",
        help="",
        default="translate English to German: That is good.",
    )
    parser.add_argument(
        "--encode-only",
        action="store_true",
        default=False,
        help="Whether to decode or not",
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

    config = ModelArgs()
    model, tokenizer = load_model(config)

    prompt = tokenizer(
        args.prompt,
        return_tensors="np",
        return_attention_mask=False,
    )["input_ids"]

    prompt = mx.array(prompt)

    if args.encode_only:
        print("[INFO] Encoding with T5...", flush=True)
        print(args.prompt, flush=True)
        embeddings = model.wte(prompt)
        encoder_output = model.encoder(embeddings, mask=None)
        print(encoder_output, flush=True)
        exit(0)

    print("[INFO] Generating with T5...", flush=True)
    print("Input: ", args.prompt, flush=True)

    decoder_inputs = mx.array([config.decoder_start_token_id])

    start = perf_counter_ns()

    tokens = []
    for token, n_tokens in zip(
        generate(prompt, decoder_inputs, model, args.temp), range(args.max_tokens)
    ):
        if token.item() == tokenizer.eos_token_id:
            break
        tokens.append(token.item())
    # For some reason using the following line doesn't give spaces
    #        print(tokenizer.decode(token.item(), clean_up_tokenization_spaces=False), end="", flush=True)
    print(tokenizer.decode(tokens), end="", flush=True)

    end = perf_counter_ns()
    elapsed = (end - start) / 1.0e9
    print()
    print(f"Time: {elapsed:.2f} seconds, tokens/s: {n_tokens / elapsed:.2f}")
