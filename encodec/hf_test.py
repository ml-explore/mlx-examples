import mlx.core as mx
import numpy as np
import torch
from transformers import AutoProcessor, EncodecModel
from utils import load, load_audio

# processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
# audio_sample = load_audio("ls_test.flac", processor.sampling_rate)


def compare_models():
    pt_model = EncodecModel.from_pretrained("facebook/encodec_24khz")
    mx_model = load("mlx_models")

    np.random.seed(0)
    audio = np.random.uniform(size=(1, 159960)).astype(np.float32)
    mask = np.ones(audio.shape, dtype=np.int32)
    pt_encoded = pt_model.encode(torch.tensor(audio)[None], torch.tensor(mask)[None])
    mx_encoded = mx_model.encode(mx.array(audio)[..., None], mx.array(mask)[..., None])
    pt_codes = pt_encoded.audio_codes.numpy()
    mx_codes = mx_encoded[0]
    assert np.array_equal(pt_codes, mx_codes), "Encoding codes mismatch"

    for mx_scale, pt_scale in zip(mx_encoded[1], pt_encoded.audio_scales):
        if mx_scale is not None:
            pt_scales = pt_scale.numpy()
            assert np.allclose(pt_scales, mx_scales, atol=1e-3, rtol=1e-4)

    pt_audio = pt_model.decode(
        pt_encoded.audio_codes, pt_encoded.audio_scales, torch.tensor(mask)[None]
    )
    pt_audio = pt_audio[0].squeeze().detach().numpy()
    mx_audio = mx_model.decode(*mx_encoded, mx.array(mask)[..., None])
    mx_audio = mx_audio.squeeze()
    assert np.allclose(
        pt_audio, mx_audio, atol=1e-5, rtol=1e-5
    ), "Decoding audio mismatch"


# pre-process the inputs
# inputs = processor(raw_audio=np.array(audio_sample), sampling_rate=processor.sampling_rate, return_tensors="pt")
# print(inputs["input_values"].shape)
# print(inputs["padding_mask"])
if __name__ == "__main__":
    compare_models()
