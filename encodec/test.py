# Copyright © 2024 Apple Inc.

import mlx.core as mx
import numpy as np
import torch
from transformers import AutoProcessor
from transformers import EncodecModel as PTEncodecModel

from encodec import EncodecModel, preprocess_audio


def compare_processors():
    np.random.seed(0)
    audio_length = 95500
    audio = np.random.uniform(size=(2, audio_length)).astype(np.float32)

    processor = AutoProcessor.from_pretrained("facebook/encodec_48khz")

    pt_inputs = processor(
        raw_audio=audio, sampling_rate=processor.sampling_rate, return_tensors="pt"
    )
    mx_inputs = preprocess_audio(
        mx.array(audio).T,
        processor.sampling_rate,
        processor.chunk_length,
        processor.chunk_stride,
    )

    assert np.array_equal(pt_inputs["input_values"], mx_inputs[0].moveaxis(2, 1))
    assert np.array_equal(pt_inputs["padding_mask"], mx_inputs[1])


def compare_models():
    pt_model = PTEncodecModel.from_pretrained("facebook/encodec_48khz")
    mx_model, _ = EncodecModel.from_pretrained("mlx-community/encodec-48khz-float32")

    np.random.seed(0)
    audio_length = 190560
    audio = np.random.uniform(size=(1, audio_length, 2)).astype(np.float32)
    mask = np.ones((1, audio_length), dtype=np.int32)
    pt_encoded = pt_model.encode(
        torch.tensor(audio).moveaxis(2, 1), torch.tensor(mask)[None]
    )
    mx_encoded = mx_model.encode(mx.array(audio), mx.array(mask))
    pt_codes = pt_encoded.audio_codes.numpy()
    mx_codes = mx_encoded[0]
    assert np.array_equal(pt_codes, mx_codes), "Encoding codes mismatch"

    for mx_scale, pt_scale in zip(mx_encoded[1], pt_encoded.audio_scales):
        if mx_scale is not None:
            pt_scale = pt_scale.numpy()
            assert np.allclose(pt_scale, mx_scale, atol=1e-3, rtol=1e-4)

    pt_audio = pt_model.decode(
        pt_encoded.audio_codes, pt_encoded.audio_scales, torch.tensor(mask)[None]
    )
    pt_audio = pt_audio[0].squeeze().T.detach().numpy()
    mx_audio = mx_model.decode(*mx_encoded, mx.array(mask))
    mx_audio = mx_audio.squeeze()
    assert np.allclose(
        pt_audio, mx_audio, atol=1e-4, rtol=1e-4
    ), "Decoding audio mismatch"


if __name__ == "__main__":
    compare_processors()
    compare_models()
