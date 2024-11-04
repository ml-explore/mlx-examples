# Copyright © 2024 Apple Inc.

import time

import mlx.core as mx

from encodec import EncodecModel

model, processor = EncodecModel.from_pretrained("mlx-community/encodec-48khz-float32")

audio = mx.random.uniform(shape=(288000, 2))
feats, mask = processor(audio)
mx.eval(model, feats, mask)


@mx.compile
def fun():
    codes, scales = model.encode(feats, mask, bandwidth=3)
    reconstructed = model.decode(codes, scales, mask)
    return reconstructed


for _ in range(5):
    mx.eval(fun())

tic = time.time()
for _ in range(10):
    mx.eval(fun())
toc = time.time()
ms = 1000 * (toc - tic) / 10
print(f"Time per it: {ms:.3f}")
