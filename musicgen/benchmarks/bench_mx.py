import sys
import time
from pathlib import Path

import mlx.core as mx
from utils import load

text = "folk ballad"
model = load("facebook/musicgen-medium")

max_steps = 100

audio = model.generate(text, max_steps=10)
mx.eval(audio)

tic = time.time()
audio = model.generate(text, max_steps=max_steps)
mx.eval(audio)
toc = time.time()

ms = 1000 * (toc - tic) / max_steps
print(f"Time (ms) per step: {ms:.3f}")
