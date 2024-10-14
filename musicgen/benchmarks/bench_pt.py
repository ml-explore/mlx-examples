# Copyright © 2024 Apple Inc.

import time

import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

model_name = "facebook/musicgen-medium"
processor = AutoProcessor.from_pretrained(model_name)
model = MusicgenForConditionalGeneration.from_pretrained(model_name).to("mps")

inputs = processor(
    text=["folk ballad"],
    padding=True,
    return_tensors="pt",
)
inputs["input_ids"] = inputs["input_ids"].to("mps")
inputs["attention_mask"] = inputs["attention_mask"].to("mps")

# warmup
audio_values = model.generate(**inputs, max_new_tokens=10)
torch.mps.synchronize()

max_steps = 100
tic = time.time()
audio_values = model.generate(**inputs, max_new_tokens=max_steps)
torch.mps.synchronize()
toc = time.time()

ms = 1000 * (toc - tic) / max_steps
print(f"Time (ms) per step: {ms:.3f}")
