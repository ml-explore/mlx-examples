# Copyright © 2024 Apple Inc.

import time

import numpy as np
import torch
from transformers import AutoProcessor, EncodecModel

processor = AutoProcessor.from_pretrained("facebook/encodec_48khz")
audio = np.random.uniform(size=(2, 288000)).astype(np.float32)

pt_model = EncodecModel.from_pretrained("facebook/encodec_48khz").to("mps")
pt_inputs = processor(
    raw_audio=audio, sampling_rate=processor.sampling_rate, return_tensors="pt"
).to("mps")


def fun():
    pt_encoded = pt_model.encode(pt_inputs["input_values"], pt_inputs["padding_mask"])
    pt_audio = pt_model.decode(
        pt_encoded.audio_codes, pt_encoded.audio_scales, pt_inputs["padding_mask"]
    )
    torch.mps.synchronize()


for _ in range(5):
    fun()

tic = time.time()
for _ in range(10):
    fun()
toc = time.time()
ms = 1000 * (toc - tic) / 10
print(f"Time per it: {ms:.3f}")
