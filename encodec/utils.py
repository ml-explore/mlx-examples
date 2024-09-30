# Copyright © 2024 Apple Inc.

import functools
import json
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Union

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download

import encodec


def save_audio(file: str, audio: mx.array, sampling_rate: int):
    """
    Save audio to a wave (.wav) file.
    """
    from scipy.io.wavfile import write

    audio = (audio * 32767).astype(mx.int16)
    write(file, sampling_rate, np.array(audio))


def load_audio(file: str, sampling_rate: int, channels: int):
    """
    Read audio into an mx.array, resampling if necessary.

    Args:
        file (str): The audio file to open.
        sampling_rate (int): The sample rate to resample the audio at if needed.
        channels (int): The number of audio channels.

    Returns:
        An mx.array containing the audio waveform in float32.
    """
    from subprocess import CalledProcessError, run

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", str(channels),
        "-acodec", "pcm_s16le",
        "-ar", str(sampling_rate),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    out = mx.array(np.frombuffer(out, np.int16))
    return out.reshape(-1, channels).astype(mx.float32) / 32767.0


def preprocess_audio(
    raw_audio: Union[mx.array, List[mx.array]],
    sampling_rate: int = 24000,
    chunk_length: Optional[int] = None,
    chunk_stride: Optional[int] = None,
):
    r"""
    Prepare inputs for the EnCodec model.

    Args:
        raw_audio (mx.array or List[mx.array]): The sequence or batch of
            sequences to be processed.
        sampling_rate (int): The sampling rate at which the audio waveform
            should be digitalized.
        chunk_length (int, optional): The model's chunk length.
        chunk_stride (int, optional): The model's chunk stride.
    """
    if not isinstance(raw_audio, list):
        raw_audio = [raw_audio]

    raw_audio = [x[..., None] if x.ndim == 1 else x for x in raw_audio]

    max_length = max(array.shape[0] for array in raw_audio)
    if chunk_length is not None:
        max_length += chunk_length - (max_length % chunk_stride)

    inputs = []
    masks = []
    for x in raw_audio:
        length = x.shape[0]
        mask = mx.ones((length,), dtype=mx.bool_)
        difference = max_length - length
        if difference > 0:
            mask = mx.pad(mask, (0, difference))
            x = mx.pad(x, ((0, difference), (0, 0)))
        inputs.append(x)
        masks.append(mask)
    return mx.stack(inputs), mx.stack(masks)


def load(path_or_repo):
    """
    Load the model and audo preprocessor.
    """
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
    processor = functools.partial(
        preprocess_audio,
        sampling_rate=config.sampling_rate,
        chunk_length=model.chunk_length,
        chunk_stride=model.chunk_stride,
    )
    mx.eval(model)
    return model, processor
