import argparse
from typing import Optional
from dataclasses import dataclass

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
  eos_token_id: int = 1
  layer_norm_epsilon: float = 1e-06
  n_positions: int = 512
  num_heads: int = 8
  num_layers: int = 6
  decoder_start_token_id: int = 0
  pad_token_id: int = 0
  relative_attention_num_buckets: int = 32
  vocab_size: int = 32128




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
    def __init__(self, dims: int, num_heads: int, mlp_dims: Optional[int] = None):
        super().__init__()
        mlp_dims = mlp_dims or dims * 4
        self.attention = nn.MultiHeadAttention(dims, num_heads)
        self.ln1 = LayerNorm(dims)
        self.ln2 = LayerNorm(dims)
        self.linear1 = nn.Linear(dims, mlp_dims, bias=False)
        self.linear2 = nn.Linear(mlp_dims, dims, bias=False)

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
    def __init__(
        self, num_layers: int, dims: int, num_heads: int, mlp_dims: Optional[int] = None
    ):
        super().__init__()
        self.layers = [
            TransformerEncoderLayer(dims, num_heads, mlp_dims)
            for _ in range(num_layers)
        ]
        self.ln = LayerNorm(dims)

    def __call__(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln(x)

        return x


class T5(nn.Module):
    def __init__(self, config: ModelArgs):
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = TransformerEncoder(
            num_layers=config.num_layers,
            dims=config.d_model,
            num_heads=config.num_heads,
            mlp_dims=config.d_ff,
        )
        # self.decoder = TransformerDecoder(config)
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
            mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
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
    current_weights_keys = set(k for k, _ in current_weights)
    weights_to_load_keys = set(k for k, _ in weights_to_load)
    print("Missing weights: ", sorted(current_weights_keys - weights_to_load_keys))
    print()
    print("Weights ignored: ", sorted(weights_to_load_keys - current_weights_keys))
    model.update(tree_unflatten(weights_to_load))
    tokenizer = AutoTokenizer.from_pretrained("t5-small", trust_remote_code=True)
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T5 Inference script")
    parser.add_argument(
        "--prompt",
        help="translate English to German: That is good.",
        default="",
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
