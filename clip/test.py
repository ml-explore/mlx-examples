from pathlib import Path
from typing import List

import mlx.core as mx
import model
import numpy as np
import torch
import transformers
from image_processor import CLIPImageProcessor
from PIL import Image
from tokenizer import CLIPTokenizer
from transformers import AutoTokenizer
from transformers.image_processing_utils import ChannelDimension

CONVERTED_WEIGHTS_PATH = Path("weights/mlx")
TEST_CKPTS: List[str] = [
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
]


def test_image_processor(TEST_CKPT: str):
    mx_image_proc = CLIPImageProcessor.from_pretrained(TEST_CKPT)
    tf_image_proc = transformers.CLIPImageProcessor.from_pretrained(TEST_CKPT)
    image = Image.open("assets/cat.jpeg")

    mx_data = mx_image_proc([image])
    tf_data = mx.array(
        np.array(
            tf_image_proc([image], data_format=ChannelDimension.LAST)["pixel_values"]
        )
    )
    assert mx.allclose(mx_data, tf_data, atol=1e-5)


def test_text_tokenizer(TEST_CKPT: str):
    texts = ["a photo of a cat", "a photo of a dog"]
    mx_tokenizer = CLIPTokenizer.from_pretrained(TEST_CKPT)
    hf_tokenizer = AutoTokenizer.from_pretrained(TEST_CKPT)

    for txt in texts:
        assert np.array_equal(
            mx_tokenizer.tokenize(txt)[None, :],
            hf_tokenizer(txt, return_tensors="np")["input_ids"],
        )


def test_text_encoder(TEST_CKPT: str):
    mx_tokenizer = CLIPTokenizer.from_pretrained(TEST_CKPT)
    mx_clip_tenc = model.CLIPTextModel.from_pretrained(
        CONVERTED_WEIGHTS_PATH / TEST_CKPT
    )
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(TEST_CKPT)
    hf_clip = transformers.CLIPTextModel.from_pretrained(TEST_CKPT)
    texts = ["a photo of a cat", "a photo of a dog"]
    # Tokenize
    tokens_hf = hf_tokenizer(texts, return_tensors="pt")
    tokens_mx = mx_tokenizer(texts)
    # Get expected
    with torch.inference_mode():
        expected_out = hf_clip(**tokens_hf)
        expected_last_hidden = expected_out.last_hidden_state.numpy()
        expected_pooler_output = expected_out.pooler_output.numpy()
    out = mx_clip_tenc(tokens_mx)
    # Test text encoder
    assert np.allclose(out.last_hidden_state, expected_last_hidden, atol=1e-5)
    assert np.allclose(out.pooler_output, expected_pooler_output, atol=1e-5)


def test_vision_encoder(TEST_CKPT: str):
    mx_clip_venc = model.CLIPVisionModel.from_pretrained(
        CONVERTED_WEIGHTS_PATH / TEST_CKPT
    )
    hf_clp_vision = transformers.CLIPVisionModel.from_pretrained(TEST_CKPT)
    hf_processor = transformers.CLIPProcessor.from_pretrained(TEST_CKPT)
    # Load and process test image
    x = hf_processor(
        images=[Image.open("assets/dog.jpeg")], return_tensors="np"
    ).pixel_values.transpose((0, 2, 3, 1))
    x = mx.array(x)
    # Infer with HuggingFace model
    with torch.inference_mode():
        # Get expected
        x_tc = torch.tensor(x.tolist())
        x_tc = x_tc.permute((0, 3, 1, 2))
        expected_out = hf_clp_vision(x_tc)
        expected_last_hidden = expected_out.last_hidden_state.numpy()
        expected_pooler_output = expected_out.pooler_output.numpy()

    # Test MLX vision encoder
    out = mx_clip_venc(x)
    assert np.allclose(
        out.last_hidden_state, expected_last_hidden, rtol=1e-4, atol=1e-3
    )
    assert np.allclose(out.pooler_output, expected_pooler_output, rtol=1e-4, atol=1e-3)


def test_clip_model(TEST_CKPT: str):
    clip = model.CLIPModel.from_pretrained(CONVERTED_WEIGHTS_PATH / TEST_CKPT)
    hf_clip = transformers.CLIPModel.from_pretrained(TEST_CKPT)
    hf_processor = transformers.CLIPProcessor.from_pretrained(TEST_CKPT)

    clip_input = hf_processor(
        text=["a photo of a cat", "a photo of a dog"],
        images=[Image.open("assets/cat.jpeg"), Image.open("assets/dog.jpeg")],
        return_tensors="pt",
    )
    with torch.inference_mode():
        expected_out = hf_clip(**clip_input, return_loss=True)

    out = clip(
        input_ids=mx.array(clip_input.input_ids.numpy()),
        pixel_values=mx.array(clip_input.pixel_values.numpy()).transpose((0, 2, 3, 1)),
        return_loss=True,
    )

    assert np.allclose(out.text_embeds, expected_out.text_embeds, atol=1e-5)
    assert np.allclose(out.image_embeds, expected_out.image_embeds, atol=1e-5)
    assert np.allclose(out.loss, expected_out.loss, atol=1e-5)


for TEST_CKPT in TEST_CKPTS:
    print(f"[testing] {TEST_CKPT}")
    test_image_processor(TEST_CKPT)
    test_text_tokenizer(TEST_CKPT)
    test_text_encoder(TEST_CKPT)
    test_vision_encoder(TEST_CKPT)
    test_clip_model(TEST_CKPT)
