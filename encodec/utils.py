# Copyright © 2024 Apple Inc.

import mlx.core as mx
import numpy as np


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
