# Copyright © 2024 Apple Inc.

import sys
import time
from pathlib import Path

import mlx.core as mx

cur_path = Path(__file__).parents[1].resolve()
sys.path.append(str(cur_path))

from musicgen import MusicGen

text = "folk ballad"
model = MusicGen.from_pretrained("facebook/musicgen-medium")

max_steps = 100

audio = model.generate(text, max_steps=10)
mx.eval(audio)

tic = time.time()
audio = model.generate(text, max_steps=max_steps)
mx.eval(audio)
toc = time.time()

ms = 1000 * (toc - tic) / max_steps
print(f"Time (ms) per step: {ms:.3f}")
