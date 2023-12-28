from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map, tree_unflatten
from transformers import AutoModelForCausalLM, LlamaConfig


def create_additive_causal_mask(N: int, offset: int = 0, dtype: mx.Dtype = mx.float32):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    mask = linds[:, None] < rinds[None]
    mask = mask.astype(dtype) * -1e9
    return mask


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
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config

        self.n_heads: int = config.num_attention_heads
        self.n_kv_heads: int = config.num_key_value_heads
        self.repeats = self.n_heads // self.n_kv_heads
        self.head_dim = config.hidden_size // self.n_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(
            config.hidden_size, config.hidden_size // self.repeats, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.hidden_size // self.repeats, bias=False
        )
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rope = nn.RoPE(self.head_dim, traditional=False)

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
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )  # B, n_kv_heads, L, head_dim

        def repeat(a):
            a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)
            kv_size = a.shape[-1]
            return a.reshape([B, self.n_heads, -1, kv_size])

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        scores = (queries * self.scale) @ repeat(keys).transpose(0, 1, 3, 2)
        if mask is not None:
            scores += mask
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = (scores @ repeat(values)).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), (keys, values)


class FeedForward(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.dim = config.hidden_size
        self.self_attn = Attention(config=config)
        self.mlp = FeedForward(config=config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

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


class Llama(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            TransformerBlock(config=config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.reset_cache()

    def truncate_cache(self, num_to_truncate):
        if num_to_truncate <= 0:
            return
        cache_length = self.kv_cache[0][0].shape[2]
        if num_to_truncate < cache_length:
            self.kv_cache = tree_map(
                lambda x: x[:, :, :-num_to_truncate, :], self.kv_cache
            )
        else:
            self.reset_cache()

    def reset_cache(self):
        self.kv_cache = [None] * len(self.layers)

    def __call__(
        self,
        x: mx.array,
        next_tokens: int = -1,
    ):
        if self.kv_cache[0]:
            offset = self.kv_cache[0][0].shape[-2]
        else:
            offset = 0

        if x.shape[1] > 1:
            mask = create_additive_causal_mask(x.shape[1], offset)
            mask = mask.astype(self.embed_tokens.weight.dtype)
        else:
            mask = None

        x = self.embed_tokens(x)
        for idx, layer in enumerate(self.layers):
            x, self.kv_cache[idx] = layer(x, mask, cache=self.kv_cache[idx])

        if next_tokens > 0:
            x = x[:, -next_tokens:]

        x = self.norm(x)
        return self.lm_head(x)

    @classmethod
    def from_hugging_face(cls, model_path: str, quantized: bool = True):
        config = LlamaConfig.from_pretrained(model_path)
        torch_weights = AutoModelForCausalLM.from_pretrained(model_path).state_dict()
        weights = {
            k.replace("model.", ""): mx.array(v.numpy(), mx.float16)
            for k, v in torch_weights.items()
        }
        model = cls(config)
        model.update(tree_unflatten(list(weights.items())))
        # if quantization is not None:
        #    nn.QuantizedLinear.quantize_module(model, **quantization)
        mx.eval(model.parameters())
        return model
