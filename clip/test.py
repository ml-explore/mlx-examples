import mlx.core as mx
import numpy as np
import torch
from model_io import load_text_encoder, load_tokenizer
from transformers import AutoTokenizer, CLIPModel

TEST_CKPT: str = "openai/clip-vit-large-patch14"


def test_text_tokenizer():
    texts = [
        "a photo of a cat",
        "a photo of a dog"
    ]
    mx_tokenizer = load_tokenizer(TEST_CKPT)
    hf_tokenizer = AutoTokenizer.from_pretrained(TEST_CKPT)

    for txt in texts:
        assert np.array_equal(
            mx_tokenizer.tokenize(txt)[None, :],
            hf_tokenizer(txt, return_tensors="np")['input_ids']
        )


def test_text_encoder():
    mx_tokenizer = load_tokenizer(TEST_CKPT)
    mx_clip_tenc = load_text_encoder(TEST_CKPT)
    hf_tokenizer = AutoTokenizer.from_pretrained(TEST_CKPT)
    hf_clip = CLIPModel.from_pretrained(TEST_CKPT)
    texts = [
        "a photo of a cat",
        "a photo of a dog"
    ]
    # Tokenize
    tokens_hf = hf_tokenizer(texts, return_tensors="pt")
    tokens_mx = mx_tokenizer(texts)
    # Get expected
    with torch.inference_mode():
        expected_last_hidden = hf_clip.text_model(
            **tokens_hf).last_hidden_state.numpy()
    # Test text encoder
    assert np.allclose(
        mx_clip_tenc(tokens_mx),
        expected_last_hidden,
        atol=1e-5
    )


test_text_tokenizer()
test_text_encoder()
