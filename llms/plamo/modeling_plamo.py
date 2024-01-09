import glob
from pathlib import Path
from typing import Any, List, NamedTuple, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_unflatten
from sentencepiece import SentencePieceProcessor
from transformers import PretrainedConfig


class DecoderInput(NamedTuple):
    hidden_states: mx.array
    position_ids: mx.array
    attention_mask: Optional[mx.array] = None
    past_key_values: Optional[List[mx.array]] = None
    output_hidden_states: Optional[bool] = False
    output_attentions: Optional[bool] = False
    use_cache: Optional[bool] = False
    gradient_checkpointing: bool = False


class DecoderOutput(NamedTuple):
    hidden_states: mx.array
    all_hidden_states: Optional[Tuple[mx.array, ...]]
    all_self_attns: Optional[Tuple[mx.array, ...]]
    next_decoder_cache: Optional[Tuple[mx.array, ...]]


class PlamoConfig(PretrainedConfig):  # type: ignore
    model_type: str = "plamo"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 13312,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = None,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000,
        rope_traditional: bool = False,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tokenizer_class: str = "PlamoTokenizer",
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        n_shared_head: int = 8,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_traditional = rope_traditional
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache

        self.n_shared_head = n_shared_head

        super().__init__(
            tokenizer_class=tokenizer_class,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.variance_epsilon = eps

    def _norm(self, x):
        return x * mx.rsqrt(x.square().mean(-1, keepdims=True) + self.variance_epsilon)

    def __call__(self, x):
        output = self._norm(x.astype(mx.float32)).astype(x.dtype)
        return self.weight * output


class Attention(nn.Module):
    def __init__(self, config: PlamoConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        head_dim = self.hidden_size // config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.rope_traditional = config.rope_traditional

        self.q_num_heads = config.num_attention_heads
        self.qk_dim = self.v_dim = head_dim
        self.k_num_heads = self.v_num_heads = int(
            np.ceil(self.q_num_heads / config.n_shared_head)
        )

        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size, self.q_num_heads * self.qk_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.k_num_heads * self.qk_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.v_num_heads * self.v_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.q_num_heads * self.v_dim, self.hidden_size, bias=False
        )
        self.rotary_emb = nn.RoPE(self.qk_dim, self.rope_traditional, self.rope_theta)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Prepare the queries, keys and values for the attention computation
        query_states = query_states.reshape(
            bsz, q_len, self.q_num_heads, self.qk_dim
        ).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(
            bsz, q_len, self.k_num_heads, self.qk_dim
        ).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(
            bsz, q_len, self.v_num_heads, self.v_dim
        ).transpose(0, 2, 1, 3)

        def _expand_kv(a):
            a = mx.concatenate(
                [mx.expand_dims(a, 2)] * self.config.n_shared_head, axis=2
            )
            return a.reshape([bsz, self.q_num_heads, q_len, -1])

        # expand shared kv
        assert self.k_num_heads == self.v_num_heads
        key_states, value_states = map(_expand_kv, (key_states, value_states))

        if cache is not None:
            key_cache, value_cache = cache
            query_states = self.rotary_emb(query_states, offset=key_cache.shape[2])
            key_states = self.rotary_emb(key_states, offset=key_cache.shape[2])
            key_states = mx.concatenate([key_cache, key_states], axis=2)
            value_states = mx.concatenate([value_cache, value_states], axis=2)
        else:
            query_states = self.rotary_emb(query_states)
            key_states = self.rotary_emb(key_states)

        scores = (query_states * self.scale) @ key_states.transpose(0, 1, 3, 2)
        if attention_mask is not None:
            scores += attention_mask
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = (scores @ value_states).transpose(0, 2, 1, 3).reshape(bsz, q_len, -1)

        return self.o_proj(output), (key_states, value_states)


class MLP(nn.Module):
    def __init__(self, config: PlamoConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.silu

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))  # type: ignore


class PlamoDecoderLayer(nn.Module):
    def __init__(self, config: PlamoConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[Any, ...]:
        # from LlamaDecoder
        residual = hidden_states

        hidden_states = self.norm(hidden_states)

        # Self Attention
        hidden_states_sa, cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            cache=cache,
        )

        # Fully Connected
        hidden_states_mlp = self.mlp(hidden_states)

        # Residual ("Parallel Layers" is used here, which is different from the normal residual connection)
        # See "GPT-NeoX-20B: An Open-Source Autoregressive Language Model" for Parallel Layers
        hidden_states = residual + hidden_states_sa + hidden_states_mlp

        return hidden_states, cache  # type: ignore


class PlamoDecoder(nn.Module):
    def __init__(self, config: PlamoConfig) -> None:
        super().__init__()
        self.layers = [
            PlamoDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]


class PlamoPreTrainedModel(nn.Module):  # type: ignore
    config_class = PlamoConfig
    _no_split_modules: List[str]
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PlamoDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def __init__(self, config: PlamoConfig):
        super().__init__()
        self.config = config

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(
        self, module: nn.Module, value: bool = False
    ) -> None:
        module.gradient_checkpointing = value  # type: ignore


class PlamoModel(PlamoPreTrainedModel):
    def __init__(self, config: PlamoConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = PlamoDecoder(config)  # type: ignore
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        # self.post_init()


class PlamoForCausalLM(PlamoPreTrainedModel):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.model = PlamoModel(config)

        self.lm_head: nn.Module = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        # Initialize weights and apply final processing
        # self.post_init()

    def __call__(self, x: mx.array) -> mx.array:
        pass

    def generate(self, x: mx.array, temp=1.0) -> mx.array:
        def sample(logits):
            if temp == 0:
                return mx.argmax(logits, axis=-1)
            else:
                return mx.random.categorical(logits * (1 / temp))

        cache = []

        # Make an additive causal mask. We will need that to process the prompt.
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.model.embed_tokens.weight.dtype)

        # First we process the prompt x the same was as in __call__ but
        # save the caches in cache
        x = self.model.embed_tokens(x)
        for layer in self.model.layers.layers:
            x, c = layer(x, mask)
            # We store the per layer cache in a simple python list
            cache.append(c)
        x = self.model.norm(x)
        # We only care about the last logits that generate the next token
        y = self.lm_head(x[:, -1])
        y = sample(y)

        # y now has size [1]
        # Since MLX is lazily evaluated nothing is computed yet.
        # Calling y.item() would force the computation to happen at
        # this point but we can also choose not to do that and let the
        # user choose when to start the computation.
        yield y

        # Now we parsed the prompt and generated the first token we
        # need to feed it back into the model and loop to generate the
        # rest.
        while True:
            # Unsqueezing the last dimension to add a sequence length
            # dimension of 1
            x = y[:, None]

            x = self.model.embed_tokens(x)
            for i in range(len(cache)):
                # We are overwriting the arrays in the cache list. When
                # the computation will happen, MLX will be discarding the
                # old cache the moment it is not needed anymore.
                x, cache[i] = self.model.layers.layers[i](x, None, cache=cache[i])
            x = self.model.norm(x)
            y = sample(self.lm_head(x[:, -1]))

            yield y


def load_model(
    model_dir_path_str: str,
) -> Tuple[PlamoForCausalLM, SentencePieceProcessor]:
    model_path = Path(model_dir_path_str)

    unsharded_weights_path = Path(model_path / "weights.npz")
    if unsharded_weights_path.is_file():
        print("[INFO] Loading model from {}.".format(unsharded_weights_path))
        weights = mx.load(str(unsharded_weights_path))
    else:
        sharded_weights_glob = str(model_path / "weights.*.npz")
        weight_files = glob.glob(sharded_weights_glob)
        print("[INFO] Loading model from {}.".format(sharded_weights_glob))

        if len(weight_files) == 0:
            raise FileNotFoundError("No weights found in {}".format(model_path))

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf).items())

    config = PlamoConfig.from_json_file(model_path / "config.json")
    model = PlamoForCausalLM(config)
    model.update(tree_unflatten(list(weights.items())))
    tokenizer = SentencePieceProcessor(model_file=str(model_path / "tokenizer.model"))

    return model, tokenizer
