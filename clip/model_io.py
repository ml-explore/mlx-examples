# Copyright © 2023 Apple Inc.

import json
import pickle
import pickletools

import mlx.core as mx
import numpy as np
from config import CLIPTextConfig
from huggingface_hub import hf_hub_download
from mlx.utils import tree_unflatten
from model import CLIPTextModel
from safetensors import safe_open as safetensor_open
from tokenizer import Tokenizer

_DEFAULT_MODEL = "openai/clip-vit-large-patch14"
_MODELS = {
    "openai/clip-vit-large-patch14": {
        "config": "config.json",
        "model": "model.safetensors",
        "tokenizer_config": "tokenizer_config.json",
        "tokenizer_vocab": "vocab.json",
        "tokenizer_merges": "merges.txt",
    }
}


def _from_numpy(x):
    return mx.array(np.ascontiguousarray(x))


def map_clip_text_encoder_weights(key, value):
    # Remove prefixes
    if key.startswith("text_model."):
        key = key[11:]
    if key.startswith("embeddings."):
        key = key[11:]
    if key.startswith("encoder."):
        key = key[8:]

    # Map attention layers
    if "self_attn." in key:
        key = key.replace("self_attn.", "attention.")
    if "q_proj." in key:
        key = key.replace("q_proj.", "query_proj.")
    if "k_proj." in key:
        key = key.replace("k_proj.", "key_proj.")
    if "v_proj." in key:
        key = key.replace("v_proj.", "value_proj.")

    # Map ffn layers
    if "mlp.fc1" in key:
        key = key.replace("mlp.fc1", "linear1")
    if "mlp.fc2" in key:
        key = key.replace("mlp.fc2", "linear2")

    return [(key, _from_numpy(value))]


def _flatten(params):
    return [(k, v) for p in params for (k, v) in p]


def _load_safetensor_weights(mapper, model, weight_file, float16: bool = False):
    dtype = np.float16 if float16 else np.float32
    with safetensor_open(weight_file, framework="numpy") as f:
        weights = _flatten(
            [mapper(k, f.get_tensor(k).astype(dtype)) for k in f.keys()])
    model.update(tree_unflatten(weights))


def _check_key(key: str, part: str):
    if key not in _MODELS:
        raise ValueError(
            f"[{part}] '{key}' model not found, choose one of {{{','.join(_MODELS.keys())}}}"
        )


def load_text_encoder(key: str = _DEFAULT_MODEL, float16: bool = False):

    # Download the config and create the model
    config = _MODELS[key]["config"]
    with open(hf_hub_download(key, config)) as f:
        config = json.load(f)

    text_config = config["text_config"]

    model = CLIPTextModel(
        CLIPTextConfig(
            num_hidden_layers=text_config["num_hidden_layers"],
            hidden_size=text_config["hidden_size"],
            intermediate_size=text_config["intermediate_size"],
            projection_dim=text_config["projection_dim"],
            num_attention_heads=text_config["num_attention_heads"],
            max_position_embeddings=text_config["max_position_embeddings"],
            vocab_size=text_config["vocab_size"]
        )
    )

    model_weights_path = hf_hub_download(key,  _MODELS[key]["model"])
    _load_safetensor_weights(
        map_clip_text_encoder_weights, model, model_weights_path)

    return model


def load_tokenizer(key: str = _DEFAULT_MODEL):
    _check_key(key, "load_tokenizer")

    vocab_file = hf_hub_download(key, _MODELS[key]["tokenizer_vocab"])
    with open(vocab_file, encoding="utf-8") as f:
        vocab = json.load(f)

    merges_file = hf_hub_download(key, _MODELS[key]["tokenizer_merges"])
    with open(merges_file, encoding="utf-8") as f:
        bpe_merges = f.read().strip().split("\n")[1: 49152 - 256 - 2 + 1]
    bpe_merges = [tuple(m.split()) for m in bpe_merges]
    bpe_ranks = dict(map(reversed, enumerate(bpe_merges)))

    return Tokenizer(bpe_ranks, vocab)
