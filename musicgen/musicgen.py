# Copyright © 2024 Apple Inc.

import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from tqdm import tqdm

from encodec import EncodecModel
from t5 import T5


class TextConditioner(nn.Module):
    def __init__(self, t5_name, input_dim, output_dim):
        super().__init__()
        self._t5, self.tokenizer = T5.from_pretrained(t5_name)
        self.output_proj = nn.Linear(input_dim, output_dim)

    def __call__(self, text):
        x = self.tokenizer.encode(text)
        x = self._t5.encode(x)
        return self.output_proj(x)


class KVCache:
    def __init__(self, head_dim, n_kv_heads):
        self.n_kv_heads = n_kv_heads
        if isinstance(head_dim, int):
            self.k_head_dim = self.v_head_dim = head_dim
        elif isinstance(head_dim, tuple) and len(head_dim) == 2:
            self.k_head_dim, self.v_head_dim = head_dim
        else:
            raise ValueError("head_dim must be an int or a tuple of two ints")
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = 256

    def update_and_fetch(self, keys, values):
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B = keys.shape[0]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, self.n_kv_heads, n_steps * self.step, self.k_head_dim)
            v_shape = (B, self.n_kv_heads, n_steps * self.step, self.v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    @property
    def state(self):
        return self.keys, self.values


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()

        self.n_heads = n_heads

        head_dim = dim // n_heads

        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def __call__(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L_q, D = queries.shape
        L_k = keys.shape[1]

        queries, keys, values = (
            self.q_proj(queries),
            self.k_proj(keys),
            self.v_proj(values),
        )

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L_q, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L_k, self.n_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L_k, self.n_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L_q, -1)
        return self.out_proj(output)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.decoder.num_attention_heads
        self.hidden_size = config.decoder.hidden_size
        self.self_attn = MultiHeadAttention(self.hidden_size, self.num_attention_heads)
        self.cross_attn = MultiHeadAttention(self.hidden_size, self.num_attention_heads)
        self.linear1 = nn.Linear(self.hidden_size, config.decoder.ffn_dim, bias=False)
        self.linear2 = nn.Linear(config.decoder.ffn_dim, self.hidden_size, bias=False)

        self.norm1 = nn.LayerNorm(self.hidden_size, eps=1e-5)
        self.norm_cross = nn.LayerNorm(self.hidden_size, eps=1e-5)
        self.norm2 = nn.LayerNorm(self.hidden_size, eps=1e-5)

    def __call__(
        self,
        x: mx.array,
        conditioning: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        xn = self.norm1(x)
        x += self.self_attn(xn, xn, xn, mask, cache)
        xn = self.norm_cross(x)
        x += self.cross_attn(xn, conditioning, conditioning, mask)
        xn = self.norm2(x)
        x += self.linear2(nn.gelu(self.linear1(xn)))
        return x


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def top_k_sampling(
    logits: mx.array, top_k: float, temperature: float, axis: int = -1
) -> mx.array:
    """
    Apply top-k sampling to logits.

    Args:
        logits: The logits from the model's output.
        top_k: Sample from the top k logits.
        temperature: Temperature parameter for softmax distribution reshaping.
        axis: Axis along which to sample.
    Returns:
        token selected based on the top-k criterion.
    """
    # referenced implementation from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L449-L460
    probs = mx.softmax(logits * (1 / temperature), axis=axis)

    # sort probs in ascending order
    sorted_indices = mx.argsort(probs, axis=axis)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=axis)
    prob_threshold = mx.take(sorted_probs, mx.array(-top_k), axis=axis)

    # select the top K tokens in probability
    top_probs = mx.where(
        sorted_probs > prob_threshold,
        sorted_probs,
        0,
    )

    sorted_token = mx.random.categorical(mx.log(top_probs), axis=axis)
    token = mx.take_along_axis(
        sorted_indices, mx.expand_dims(sorted_token, axis), axis=axis
    )

    return token


def create_sin_embedding(positions: mx.array, dim: int, max_period: float = 10000):
    assert dim % 2 == 0
    half_dim = dim // 2
    adim = mx.arange(half_dim).reshape(1, 1, -1)
    phase = positions / (max_period ** (adim / (half_dim - 1)))
    return mx.concatenate([mx.cos(phase), mx.sin(phase)], axis=-1)


class MusicGen(nn.Module):
    def __init__(self, config):
        self.num_codebooks = config.decoder.num_codebooks
        self.codebook_size = config.audio_encoder.codebook_size
        self.bos_token_id = config.decoder.bos_token_id
        self.hidden_size = config.decoder.hidden_size
        self.num_attention_heads = config.decoder.num_attention_heads
        self.sampling_rate = config.audio_encoder.sampling_rate

        self.text_conditioner = TextConditioner(
            config.text_encoder._name_or_path,
            config.text_encoder.d_model,
            self.hidden_size,
        )
        self.emb = [
            nn.Embedding(self.codebook_size + 1, self.hidden_size)
            for _ in range(self.num_codebooks)
        ]
        self.layers = [
            TransformerBlock(config) for _ in range(config.decoder.num_hidden_layers)
        ]
        self.out_norm = nn.LayerNorm(self.hidden_size, eps=1e-5)
        self.linears = [
            nn.Linear(self.hidden_size, self.codebook_size, bias=False)
            for _ in range(self.num_codebooks)
        ]
        encodec_name = config.audio_encoder._name_or_path.split("/")[-1]
        encodec_name = encodec_name.replace("_", "-")
        self._audio_decoder, _ = EncodecModel.from_pretrained(
            f"mlx-community/{encodec_name}-float32"
        )

    def __call__(
        self,
        audio_tokens: mx.array,
        conditioning: mx.array,
        cache: list[KVCache] = None,
    ):

        if cache is None:
            cache = [None] * len(self.layers)

        x = sum([self.emb[k](audio_tokens[..., k]) for k in range(self.num_codebooks)])

        offset = cache[0].offset if cache[0] is not None else 0
        pos_emb = create_sin_embedding(offset, self.hidden_size)
        x += pos_emb.astype(x.dtype)

        for layer, c in zip(self.layers, cache):
            x = layer(x, conditioning, cache=c)

        x = self.out_norm(x)
        x = mx.stack([self.linears[k](x) for k in range(self.num_codebooks)], axis=-1)
        return x

    def generate(
        self,
        text: str,
        max_steps: int = 200,
        top_k: int = 250,
        temp: float = 1.0,
        guidance_coef: float = 3.0,
    ) -> mx.array:
        """
        Generates a waveform conditioned on `text`.

        Args:
            text (str): The text to condition generation on.
            max_steps (int): Max steps to generate.
            top_k (int): Top k used in sampling.
            temp (float): Sampling softmax temperature.
            guidance_coef (float): Classifier free guidance coefficent.
                Used to combine conditional and unconditional logits.

        Returns:
            An mx.array of audio samples of shape ``(num_samples,)``.
        """
        # Assuming no audio prompt we start with all bos token for the codebooks
        audio_shape = (1, max_steps + 1, self.num_codebooks)
        audio_seq = mx.full(audio_shape, self.bos_token_id)

        text_tokens = self.text_conditioner(text)
        # Compute conditional and unconditional logits in one batch
        text_tokens = mx.concatenate([text_tokens, mx.zeros_like(text_tokens)], axis=0)

        head_dim = self.hidden_size // self.num_attention_heads
        cache = [
            KVCache(head_dim, self.num_attention_heads) for _ in range(len(self.layers))
        ]
        for offset in tqdm(range(max_steps)):
            audio_input = mx.tile(audio_seq[:, offset : offset + 1], [2, 1, 1])
            audio_logits = self(audio_input, text_tokens, cache)
            cond_logits, uncond_logits = audio_logits[:1], audio_logits[1:2]
            audio_logits = uncond_logits + (cond_logits - uncond_logits) * guidance_coef
            audio_tokens = top_k_sampling(audio_logits, top_k, temp, axis=-2)
            # "delay" pattern
            audio_tokens[..., offset + 1 :] = self.bos_token_id
            audio_tokens[..., : -max_steps + offset] = self.bos_token_id
            audio_seq[:, offset + 1 : offset + 2] = audio_tokens
            mx.eval(audio_seq)

        # Undo delay
        for i in range(self.num_codebooks):
            audio_seq[:, : -self.num_codebooks, i] = audio_seq[
                :, i : -self.num_codebooks + i, i
            ]
        audio_seq = audio_seq[:, 1 : -self.num_codebooks + 1]

        audio_seq = mx.swapaxes(audio_seq, -1, -2)[:, mx.newaxis]
        audio = self._audio_decoder.decode(audio_seq, audio_scales=[None])
        return audio[0]

    @classmethod
    def sanitize(cls, weights):
        out_weights = {}
        for k, arr in weights.items():
            if k.startswith("transformer."):
                k = k[len("transformer.") :]

            if "cross_attention" in k:
                k = k.replace("cross_attention", "cross_attn")

            if "condition_provider" in k:
                k = k.replace(
                    "condition_provider.conditioners.description", "text_conditioner"
                )

            if "in_proj_weight" in k:
                dim = arr.shape[0] // 3
                name = "in_proj_weight"
                out_weights[k.replace(name, "q_proj.weight")] = arr[:dim]
                out_weights[k.replace(name, "k_proj.weight")] = arr[dim : dim * 2]
                out_weights[k.replace(name, "v_proj.weight")] = arr[dim * 2 :]
                continue

            out_weights[k] = arr
        return out_weights

    @classmethod
    def from_pretrained(cls, path_or_repo: str):
        import torch
        from huggingface_hub import snapshot_download

        path = Path(path_or_repo)
        if not path.exists():
            path = Path(
                snapshot_download(
                    repo_id=path_or_repo,
                    allow_patterns=["*.json", "state_dict.bin"],
                )
            )

        with open(path / "config.json", "r") as f:
            config = SimpleNamespace(**json.load(f))
            config.text_encoder = SimpleNamespace(**config.text_encoder)
            config.audio_encoder = SimpleNamespace(**config.audio_encoder)
            config.decoder = SimpleNamespace(**config.decoder)

        weights = torch.load(path / "state_dict.bin", weights_only=True)["best_state"]
        weights = {k: mx.array(v) for k, v in weights.items()}
        weights = cls.sanitize(weights)

        model = MusicGen(config)
        model.load_weights(list(weights.items()))
        return model
