# Copyright © 2024 Apple Inc.

import mlx.core as mx
import numpy as np


def save_audio(file: str, audio: mx.array, sampling_rate: int):
    """
    Save audio to a wave (.wav) file, supporting both mono and stereo.
    """
    from scipy.io.wavfile import write

    # Clip and scale audio
    audio = mx.clip(audio, -1, 1)
    audio = (audio * 32767).astype(mx.int16)
    
    # Convert to numpy array
    audio_np = np.array(audio)
    
    # Handle stereo by reshaping interleaved audio
    if audio_np.shape[1] == 1:  # Single column
        # Reshape to (samples, 2) for stereo
        audio_np = audio_np.reshape(-1, 1).repeat(2, axis=1)
    
    write(file, sampling_rate, audio_np)
