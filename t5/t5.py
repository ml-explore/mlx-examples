import argparse
from dataclasses import dataclass
from mlx.utils import tree_flatten, tree_unflatten
from transformers import AutoTokenizer

import mlx.core as mx
import mlx.nn as nn


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



class T5(nn.Module):
    def __init__(self, config: ModelArgs):
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = nn.TransformerEncoder(
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
