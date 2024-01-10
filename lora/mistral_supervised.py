"""
For Mistral supervised fine-tuning

Assumes records are of the form:

{ "input": "..",
  "output": ".." }

And renders the input using the MISTRAL prompt format

"""
from supervised_lora import TrainingRecordHandler, main
MISTRAL_INSTRUCTION_PROMPT = "[INST] {prompt_input}[/INST] "

from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
import json
import time
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
    rope_theta: float = 10000


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


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


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
        self.rope = nn.RoPE(args.head_dim, traditional=True, base=args.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
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


class Mistral(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        assert self.vocab_size > 0
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


class Tokenizer:
    def __init__(self, model_path: str):
        assert Path(model_path).exists(), model_path
        self._model = SentencePieceProcessor(model_file=model_path)
        self._sep = "▁"
        assert self._model.vocab_size() == self._model.get_piece_size()

    @property
    def eos_id(self) -> int:
        return self._model.eos_id()

    @property
    def pad_id(self) -> int:
        return self._model.pad_id()

    def encode(self, s: str, eos: bool = False) -> List[int]:
        toks = [self._model.bos_id(), *self._model.encode(s)]
        if eos:
            toks.append(self.eos_id)
        return toks

    def decode(self, t: List[int]) -> str:
        out = self._model.decode(t)
        if t and self._model.id_to_piece(t[0])[0] == self._sep:
            return " " + out
        return out


def load_model(folder: str):
    model_path = Path(folder)
    tokenizer = Tokenizer(str(model_path / "tokenizer.model"))
    with open(model_path / "config.json", "r") as f:
        config = json.loads(f.read())
        config.pop("sliding_window", None)
        config.pop("model_type", None)
        quantization = config.pop("quantization", None)
        model_args = ModelArgs(**config)
    weights = mx.load(str(model_path / "weights.npz"))
    weights = tree_unflatten(list(weights.items()))
    model = Mistral(model_args)
    if quantization is not None:
        nn.QuantizedLinear.quantize_module(model, **quantization)
    model.update(weights)
    mx.eval(model.parameters())
    return model, tokenizer


class MistralTrainingRecordHandler(TrainingRecordHandler):
    def get_input(self, record) -> str:
        return MISTRAL_INSTRUCTION_PROMPT.format(prompt_input=record["input"])

    def get_output(self, record) -> str:
        return record["output"]

def generate(prompt: mx.array, model: Mistral, temp: Optional[float] = 0.0):
    def sample(logits):
        if temp == 0:
            return mx.argmax(logits, axis=-1)
        else:
            return mx.random.categorical(logits * (1 / temp))

    logits, cache = model(prompt[None])
    y = sample(logits[:, -1, :])
    yield y

    while True:
        logits, cache = model(y[:, None], cache)
        y = sample(logits.squeeze(1))
        yield y


def generate_outer(model, prompt, tokenizer, max_tokens, tokens_per_eval, temp):
    print("[INFO] Starting generation...")
    tic = time.time()
    print(prompt, end="", flush=True)
    prompt = mx.array(tokenizer.encode(prompt))
    tokens = []
    for token, ntoks in zip(generate(prompt, model, temp), range(max_tokens)):
        tokens.append(token)
        if ntoks == 0:
            mx.eval(tokens)
            toc = time.time()
            prompt_tps = prompt.size / (toc - tic)
            tic = time.time()

        if (len(tokens) % tokens_per_eval) == 0:
            mx.eval(tokens)
            s = tokenizer.decode([t.item() for t in tokens])
            print(s, end="", flush=True)
            tokens = []

    mx.eval(tokens)
    s = tokenizer.decode([t.item() for t in tokens])
    print(s, flush=True)
    print("------")
    generation_tps = ntoks / (time.time() - tic)
    print(
        f"Tokens per second: prompt {prompt_tps:.3f}, "
        f"generation {generation_tps:.3f}"
    )


if __name__ == "__main__":
    main(MistralTrainingRecordHandler(), load_model, generate_outer)
