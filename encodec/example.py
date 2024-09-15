# Copyright © 2024 Apple Inc.

import mlx.core as mx
from utils import load, load_audio, save_audio

model, processor = load("mlx-community/encodec-48khz-float32")

audio = load_audio("fmi_0_gt.wav", model.sampling_rate, model.channels)

feats, mask = processor(audio)

codes, scales = model.encode(feats, mask, bandwidth=3)
reconstructed = model.decode(codes, scales, mask)

# Postprocess and save:
reconstructed = reconstructed[0, : len(audio)]
reconstructed = (reconstructed * 32767).astype(mx.int16)
save_audio("reconstructed.wav", reconstructed, model.sampling_rate)
