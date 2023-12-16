import argparse
import math
from dataclasses import dataclass

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
  num_heads: int = 8
  num_layers: int = 6
  decoder_start_token_id: int = 0
  eos_token_id: int = 1
  pad_token_id: int = 0
  vocab_size: int = 32128


def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
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
        relative_position = -mx.min(relative_position, mx.zeros_like(relative_position))
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    print("relative_position", relative_position)
    relative_position_if_large = max_exact + (
        mx.log(relative_position.astype(mx.float32) / max_exact)
        / np.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).astype(mx.int16)
    relative_position_if_large = mx.minimum(
        relative_position_if_large, num_buckets - 1
    )
    relative_buckets += mx.where(is_small, relative_position, relative_position_if_large)
    return relative_buckets


class RelativePositionBias(nn.Module):
    def __init__(self, config: ModelArgs, is_decoder: bool = False):
        self.bidirectional = not is_decoder
        self.num_buckets = config.relative_attention_num_buckets
        self.max_distance = config.n_positions
        self.n_heads = config.num_heads
        self.embeddings = nn.Embedding(
            config.relative_attention_num_buckets,
            config.num_heads)

    def __call__(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = mx.arange(query_length, dtype=mx.int32)[:, None]
        memory_position = mx.arange(key_length, dtype=mx.int32)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = _relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        values = self.embeddings(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = mx.expand_dims(values.transpose(2, 0, 1), 0)  # shape (1, num_heads, query_length, key_length)
        return values


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelArgs, has_relative_attention_bias: bool = False):
        super().__init__()
        self.num_heads = config.num_heads
        self.query_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.key_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.value_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.has_relative_attention_bias = has_relative_attention_bias
        if has_relative_attention_bias:
            self.relative_attention_bias = RelativePositionBias(config)

    def __call__(self, queries, keys, values, mask=None):
        queries = self.query_proj(queries)
        keys = self.key_proj(keys)
        values = self.value_proj(values)

        num_heads = self.num_heads
        B, L, D = queries.shape
        _, S, _ = keys.shape
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, num_heads, -1).transpose(0, 2, 3, 1)
        values = values.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)

        # Dimensions are [batch x num heads x sequence x hidden dim]
        scale = math.sqrt(1 / queries.shape[-1])
        scores = (queries * scale) @ keys
        if mask is not None:
            scores = scores + mask.astype(scores.dtype)

        if self.has_relative_attention_bias:
            position_bias = self.relative_attention_bias(L, S)
            scores += position_bias

        scores = mx.softmax(scores, axis=-1)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.out_proj(values_hat)

    @staticmethod
    def create_additive_causal_mask(N: int, dtype: mx.Dtype = mx.float32):
        indices = mx.arange(N)
        mask = indices[:, None] < indices[None]
        # usually inf but 1e9 is as good and softmax(full(1e9)) != nan
        # TODO: Should replace this with finfo(dtype).min
        mask = mask.astype(dtype) * -1e9
        return mask


class LayerNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        if affine:
            self.weight = mx.ones((dims,))
        self.eps = eps
        self.dims = dims

    def _extra_repr(self):
        return f"{self.dims}, eps={self.eps}, affine={'weight' in self}"

    def __call__(self, x):
        means = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - means) * mx.rsqrt(var + self.eps)
        return (self.weight * x) if "weight" in self else x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, has_relative_attention_bias: bool = False):
        super().__init__()
        mlp_dims = config.d_ff or config.d_model * 4
        self.attention = MultiHeadAttention(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        self.ln1 = LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.ln2 = LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.linear1 = nn.Linear(config.d_model, mlp_dims, bias=False)
        self.linear2 = nn.Linear(mlp_dims, config.d_model, bias=False)

    def __call__(self, x, mask):
        y = self.ln1(x)
        y = self.attention(y, y, y, mask)
        x = x + y

        y = self.ln2(x)
        y = self.linear1(y)
        y = mx.maximum(y, 0)
        y = self.linear2(y)
        x = x + y

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.layers = [
            TransformerEncoderLayer(config, has_relative_attention_bias=i == 0)
            for i in range(config.num_layers)
        ]
        self.ln = LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def __call__(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln(x)

        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, has_relative_attention_bias: bool = False):
        super().__init__()
        mlp_dims = config.d_ff or config.d_model * 4
        self.self_attention = MultiHeadAttention(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        self.cross_attention = MultiHeadAttention(config)
        self.ln1 = LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.ln2 = LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.ln3 = LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.linear1 = nn.Linear(config.d_model, mlp_dims, bias=False)
        self.linear2 = nn.Linear(mlp_dims, config.d_model, bias=False)

    def __call__(self, x, memory, x_mask, memory_mask):
        y = self.ln1(x)
        y = self.self_attention(y, y, y, x_mask)
        x = x + y

        y = self.ln2(x)
        y = self.cross_attention(x, memory, memory, memory_mask)
        x = x + y

        y = self.ln3(x)
        y = self.linear1(y)
        y = mx.maximum(y, 0)
        y = self.linear2(y)
        x = x + y

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.layers = [
            TransformerDecoderLayer(config, has_relative_attention_bias=i == 0)
            for i in range(config.num_layers)
        ]
        self.ln = LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def __call__(self, x, memory, x_mask, memory_mask):
        for layer in self.layers:
            x = layer(x, memory, x_mask, memory_mask)
        x = self.ln(x)

        return x


class T5(nn.Module):
    def __init__(self, config: ModelArgs):
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        # self.lm_head = OutputHead(config)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache: mx.array = None,
    ) -> tuple[mx.array, mx.array]:
        x = self.wte(inputs)

        mask = None
        if x.shape[1] > 1:
            mask = MultiHeadAttention.create_additive_causal_mask(x.shape[1])
            mask = mask.astype(x.dtype)

        y = self.encoder(x, mask)  #, cache)
        # y, cache = self.decoder(x, mask, cache)
        # return self.lm_head(y), cache
        return y  #, cache


# def generate(prompt: mx.array, model: T5, temp: Optional[float] = 0.0):
#     def sample(logits):
#         if temp == 0:
#             return mx.argmax(logits, axis=-1)
#         else:
#             return mx.random.categorical(logits * (1 / temp))

#     logits, cache = model(prompt)
#     y = sample(logits[:, -1, :])
#     yield y

#     while True:
#         logits, cache = model(y[:, None], cache=cache)
#         y = sample(logits.squeeze(1))
#         yield y


def load_model():
    model = T5(ModelArgs())
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

    model, tokenizer = load_model()

    prompt = tokenizer(
        args.prompt,
        return_tensors="np",
        return_attention_mask=False,
    )["input_ids"]

    prompt = mx.array(prompt)

    print("[INFO] Generating with T5...", flush=True)
    print(args.prompt, end="", flush=True)

    print(model(prompt))

    # tokens = []
    # for token, _ in zip(generate(prompt, model), range(args.max_tokens)):
    #     tokens.append(token)

    #     if (len(tokens) % 10) == 0:
    #         mx.eval(tokens)
    #         s = tokenizer.decode([t.item() for t in tokens])
    #         print(s, end="", flush=True)
    #         tokens = []

    # mx.eval(tokens)
    # s = tokenizer.decode([t.item() for t in tokens])
    # print(s, flush=True)
