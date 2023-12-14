from typing import Optional
from dataclasses import dataclass
from mlx.utils import tree_unflatten, tree_map
from transformers import AutoTokenizer

import mlx.core as mx
import mlx.nn as nn
import math


@dataclass
class ModelArgs:
    max_sequence_length: int = 2048
    num_vocab: int = 51200
    model_dim: int = 2560
    num_heads: int = 32
    num_layers: int = 32
    rotary_dim: int = 32


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __call__(self, input: mx.array) -> mx.array:
        return (
            0.5
            * input
            * (
                1.0
                + mx.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * (input**3)))
            )
        )


class RoPEAttention(nn.Module):
    def __init__(self, dims: int, num_heads: int, bias: bool = True):
        super().__init__()

        self.num_heads = num_heads

        self.rope = nn.RoPE(dims // num_heads, traditional=True)
        self.query_proj = nn.Linear(dims, dims, bias=bias)
        self.key_proj = nn.Linear(dims, dims, bias=bias)
        self.value_proj = nn.Linear(dims, dims, bias=bias)
        self.out_proj = nn.Linear(dims, dims, bias=bias)

    def __call__(self, queries, keys, values, mask=None, cache=None):
        queries = self.query_proj(queries)
        keys = self.key_proj(keys)
        values = self.value_proj(values)

        # Extract some shapes
        num_heads = self.num_heads
        B, L, D = queries.shape

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)

        # Add RoPE to the queries and keys and combine them with the cache
        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Finally perform the attention computation
        scale = math.sqrt(1 / queries.shape[-1])
        scores = (queries * scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores = scores + mask
        scores = mx.softmax(scores, axis=-1)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        # Note that we return the keys and values to possibly be used as a cache
        return self.out_proj(values_hat), (keys, values)


class ParallelBlock(nn.Module):
    def __init__(self, dims: int, num_heads: int, mlp_dims: Optional[int] = None):
        super().__init__()
        mlp_dims = mlp_dims or dims * 4
        self.self_attention = RoPEAttention(dims, num_heads, bias=True)
        self.ln = nn.LayerNorm(dims)
        self.fc1 = nn.Linear(dims, mlp_dims)
        self.fc2 = nn.Linear(mlp_dims, dims)
        self.act = NewGELUActivation()

    def __call__(self, x, x_mask):
        residual = x
        hidden_states = self.ln(x)
        attn_outputs, _ = self.self_attention(
            hidden_states, hidden_states, hidden_states, x_mask
        )
        ff_hidden_states = self.fc2(self.act(self.fc1(hidden_states)))

        hidden_states = attn_outputs + ff_hidden_states + residual

        return hidden_states


class TransformerDecoder(nn.Module):
    def __init__(
        self, num_layers: int, dims: int, num_heads: int, mlp_dims: Optional[int] = None
    ):
        super().__init__()
        self.h = [ParallelBlock(dims, num_heads, mlp_dims) for i in range(num_layers)]

    def __call__(self, x, x_mask):
        for layer in self.h:
            x = layer(x, x_mask)
        return x


class Phi2(nn.Module):
    def __init__(self, config: ModelArgs):
        self.wte = nn.Embedding(config.num_vocab, config.model_dim)
        self.transformer = TransformerDecoder(
            num_layers=config.num_layers,
            dims=config.model_dim,
            num_heads=config.num_heads,
        )

        self.lm_head = LanguageModelingHead(config)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array = None,
    ) -> tuple[mx.array, mx.array]:
        x = self.wte(input_ids)

        if attention_mask is not None:
            # convert 0's to -infs, 1's to 0's, and make it broadcastable
            attention_mask = mx.log(attention_mask)
            attention_mask = mx.expand_dims(attention_mask, (1, 2))
        else:
            attention_mask = nn.MultiHeadAttention.create_additive_causal_mask(
                x.shape[1]
            )

        y = self.transformer(x, attention_mask)
        return self.lm_head(y)

    def generate(self, input_ids, temp=1.0):
        cache = input_ids.tolist()

        # Make an additive causal mask. We will need that to process the prompt.
        mask = nn.MultiHeadAttention.create_additive_causal_mask(input_ids.shape[1])
        mask = mask.astype(self.wte.weight.dtype)

        # First we process the prompt x the same way as in __call__ but
        # save the caches in cache
        x = self.wte(input_ids)
        # for l in self.layers:
        #    x, c = l(x, mask=mask)
        # cache.append(c)  # <--- we store the per layer cache in a
        #      simple python list
        x = self.transformer(x, mask)
        y = self.lm_head(x[:, -1])  # <--- we only care about the last logits
        #      that generate the next token
        y = mx.random.categorical(y * (1 / temp))

        # y now has size [1]
        # Since MLX is lazily evaluated nothing is computed yet.
        # Calling y.item() would force the computation to happen at
        # this point but we can also choose not to do that and let the
        # user choose when to start the computation.
        yield y
        cache += [y.item()]

        # Now we parsed the prompt and generated the first token we
        # need to feed it back into the model and loop to generate the
        # rest.
        while True:
            # Unsqueezing the last dimension to add a sequence length
            # dimension of 1
            x = self.wte(mx.array(cache))
            x = self.transformer(x, mask)
            y = self.lm_head(x[:, -1])
            y = mx.random.categorical(y * (1 / temp))
            cache += [y[0].item()]

            yield y


class LanguageModelingHead(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        self.ln = nn.LayerNorm(config.model_dim)
        self.linear = nn.Linear(config.model_dim, config.num_vocab)

    def __call__(self, inputs):
        return self.linear(self.ln(inputs))


if __name__ == "__main__":
    model = Phi2(ModelArgs())

    weights = mx.load("weights/phi-2.npz")
    weights = tree_unflatten(list(weights.items()))
    weights = tree_map(lambda p: mx.array(p), weights)

    model.update(weights)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokens = tokenizer(
        '''def print_prime(n):
   """
   Print all primes between 1 and n
   """''',
        return_tensors="np",
        return_attention_mask=False,
    )

    tokens = {key: mx.array(v) for key, v in tokens.items()}

    print(
        '''def print_prime(n):
   """
   Print all primes between 1 and n
   """'''
    )
    for output in model.generate(**tokens):
        print(tokenizer.decode(output.item()))
