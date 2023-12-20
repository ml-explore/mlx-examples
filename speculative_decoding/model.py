from transformers import LlamaConfig, AutoModelForCausalLM
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten, tree_map
import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple

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
        # print("heads", self.n_heads, "kv heads", self.n_kv_heads, "repeats", self.repeats)
        self.head_dim = config.hidden_size // self.n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size // self.repeats, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size // self.repeats, bias=False)
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
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3) # B, n_kv_heads, L, head_dim

        def repeat(a):
            a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)
            kv_size = a.shape[-1]
            # can't use the L from x here, this is like cross-attention during decoding
            return a.reshape([B, self.n_heads, -1, kv_size])

        # cache should be with unrepeated kv, otherwise GQA is pointless lol
        # keys, values = map(repeat, (keys, values))

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # print("queries shape", queries.shape, "keys shape", keys.shape, "values shape", values.shape)

        scores = (queries * self.scale) @ repeat(keys).transpose(0, 1, 3, 2)
        if mask is not None:
            # print("we need to add mask of shape", mask.shape, "to scores of shape", scores.shape)
            if cache is None:
                scores += mask
            else:
                # we're doing "cross-attn"; add mask to the "end" of the attn matrix along the K dimension
                a, b = mx.split(scores, indices_or_sections=[-mask.shape[-1]], axis=-1)
                scores = mx.concatenate([a, b + mask], axis=-1)
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = (scores @ repeat(values)).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), (keys, values)

class FeedForward(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

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
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        self.layers = [TransformerBlock(config=config) for _ in range(config.num_hidden_layers)]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.kv_cache = []

    def truncate_kv_cache(self, num_to_truncate):
        cache_length = self.kv_cache[0][0].shape[2]
        num_to_truncate = min(num_to_truncate, cache_length)
        if num_to_truncate == 0:
            return False
        else:
            self.kv_cache = tree_map(lambda x: x[:, :, :-num_to_truncate, :], self.kv_cache)
            return True

    def __call__(
        self,
        x: mx.array, 
        read_cache: bool = False,
        write_cache: bool = False,
        next_token_only: bool = False
    ):
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.embed_tokens.weight.dtype)
        if read_cache and len(self.kv_cache) != len(self.layers):
            raise RuntimeError(f"Length of cache ({len(self.kv_cache)}) must match number of layers ({len(self.layers)})")
        x = self.embed_tokens(x)
        for idx, layer in enumerate(self.layers):
            x, c = layer(x, mask, cache=self.kv_cache[idx] if read_cache else None)
            if write_cache:
                if len(self.kv_cache) == 0:
                    self.kv_cache = [None] * len(self.layers)
                self.kv_cache[idx] = c
        x = self.norm(x)
        if next_token_only:
            x = x[:, -1]
        return self.lm_head(x)
    
    @classmethod
    def from_hugging_face(cls, model_path: str):
        config = LlamaConfig.from_pretrained(model_path)
        torch_weights = AutoModelForCausalLM.from_pretrained(model_path).state_dict()
        mx_weights = {k.replace("model.", ""):mx.array(v.numpy()) for k, v in torch_weights.items()}
        for k in mx_weights.keys():
            mx_weights[k] = mx_weights[k].astype(mx.float16)
        mlx_model = cls(config)
        mlx_model.update(tree_unflatten(list(mx_weights.items())))
            
        return mlx_model

    def generate(
        self, 
        x: mx.array, 
        temp=1.0,
        read_cache: bool = False
    ):
        # Make an additive causal mask. We will need that to process the prompt.
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.embed_tokens.weight.dtype)

        logit = self(x, read_cache=read_cache, write_cache=True, next_token_only=True)
        tok = mx.random.categorical(logit * (1 / temp))
        yield tok
        while True:
            x = tok.reshape(-1, 1)
            logit = self(x, read_cache=True, write_cache=True, next_token_only=True)
            tok = mx.random.categorical(logit * (1 / temp))
            yield tok