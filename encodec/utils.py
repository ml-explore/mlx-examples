# Copyright © 2024 Apple Inc.

import json
from pathlib import Path
from subprocess import CalledProcessError, run
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download

import encodec


def load_audio(file: str, sr: int):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Args:
        file (str): The audio file to open.
        sr (int): The sample rate to resample the audio at, if needed.

    Returns:
        An mx.array containing the audio waveform in float32.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return mx.array(np.frombuffer(out, np.int16)).flatten().astype(mx.float32) / 32768.0


def load(path_or_repo):
    path = Path(path_or_repo)
    if not path.exists():
        path = Path(
            snapshot_download(
                repo_id=path_or_repo,
                allow_patterns=["*.json", "*.safetensors", "*.model"],
            )
        )

    with open(path / "config.json", "r") as f:
        config = SimpleNamespace(**json.load(f))

    model = encodec.EncodecModel(config)
    model.load_weights(str(path / "model.safetensors"))
    return model
