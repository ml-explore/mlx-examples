# Copyright © 2024 Apple Inc.

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download

import musicgen


def save_audio(file: str, audio: mx.array, sampling_rate: int):
    """
    Save audio to a wave (.wav) file.
    """
    from scipy.io.wavfile import write

    audio = mx.clip(audio, -1, 1)
    audio = (audio * 32767).astype(mx.int16)
    write(file, sampling_rate, np.array(audio))


def load(path_or_repo):
    """
    Load the model and audio preprocessor.
    """
    import torch

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
    weights = {k: mx.array(v.numpy()) for k, v in weights.items()}

    decoder_weights = {}
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
            decoder_weights[k.replace(name, "q_proj.weight")] = arr[:dim]
            decoder_weights[k.replace(name, "k_proj.weight")] = arr[dim : dim * 2]
            decoder_weights[k.replace(name, "v_proj.weight")] = arr[dim * 2 :]
            continue

        decoder_weights[k] = arr

    model = musicgen.MusicGen(config)
    model.load_weights(list(decoder_weights.items()))
    mx.eval(model)
    return model
