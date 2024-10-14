# Copyright © 2024 Apple Inc.

import mlx.core as mx
from utils import load_audio, save_audio

from encodec import EncodecModel

# Load the 48 KHz model and preprocessor.
model, processor = EncodecModel.from_pretrained("mlx-community/encodec-48khz-float32")

# Load an audio file
audio = load_audio("/path/to/audio", model.sampling_rate, model.channels)

# Preprocess the audio (this can also be a list of arrays for batched
# processing).
feats, mask = processor(audio)


# Encode at the given bandwidth. A lower bandwidth results in more
# compression but lower reconstruction quality.
@mx.compile
def encode(feats, mask):
    return model.encode(feats, mask, bandwidth=3)


# Decode to reconstruct the audio
@mx.compile
def decode(codes, scales, mask):
    return model.decode(codes, scales, mask)


codes, scales = encode(feats, mask)
reconstructed = decode(codes, scales, mask)

# Trim any padding:
reconstructed = reconstructed[0, : len(audio)]

# Save the audio as a wave file
save_audio("reconstructed.wav", reconstructed, model.sampling_rate)
