# Copyright © 2024 Apple Inc.

import mlx.core as mx
from utils import load, load_audio, save_audio

model, processor = load("mlx-community/encodec-48khz-float32")

audio = load_audio("path/to/aduio", model.sampling_rate, model.channels)

feats, mask = processor(audio)


@mx.compile
def encode(feats, mask):
    return model.encode(feats, mask, bandwidth=3)


@mx.compile
def decode(codes, scales, mask):
    return model.decode(codes, scales, mask)


codes, scales = encode(feats, mask)
reconstructed = decode(codes, scales, mask)

# Postprocess and save:
reconstructed = reconstructed[0, : len(audio)]
reconstructed = (reconstructed * 32767).astype(mx.int16)
save_audio("reconstructed.wav", reconstructed, model.sampling_rate)
