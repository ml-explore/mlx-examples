# Copyright © 2024 Apple Inc.

import mlx.core as mx
import numpy as np


def save_audio(file: str, audio: mx.array, sampling_rate: int):
    """
    Save audio to a wave (.wav) file.
    """
    from scipy.io.wavfile import write

    audio = mx.clip(audio, -1, 1)
    audio = (audio * 32767).astype(mx.int16)
    write(file, sampling_rate, np.array(audio))
