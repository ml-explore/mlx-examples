import mlx.core as mx
import numpy as np
import torch
from model_io import load_text_encoder, load_tokenizer, load_vision_encoder
from transformers import AutoTokenizer, CLIPTextModel, CLIPVisionModel

TEST_CKPT: str = "openai/clip-vit-large-patch14"


def test_text_tokenizer():
    texts = ["a photo of a cat", "a photo of a dog"]
    mx_tokenizer = load_tokenizer(TEST_CKPT)
    hf_tokenizer = AutoTokenizer.from_pretrained(TEST_CKPT)

    for txt in texts:
        assert np.array_equal(
            mx_tokenizer.tokenize(txt)[None, :],
            hf_tokenizer(txt, return_tensors="np")["input_ids"],
        )


def test_text_encoder():
    mx_tokenizer = load_tokenizer(TEST_CKPT)
    mx_clip_tenc = load_text_encoder(TEST_CKPT)
    hf_tokenizer = AutoTokenizer.from_pretrained(TEST_CKPT)
    hf_clip = CLIPTextModel.from_pretrained(TEST_CKPT)
    texts = ["a photo of a cat", "a photo of a dog"]
    # Tokenize
    tokens_hf = hf_tokenizer(texts, return_tensors="pt")
    tokens_mx = mx_tokenizer(texts)
    # Get expected
    with torch.inference_mode():
        expected_last_hidden = hf_clip(**tokens_hf).last_hidden_state.numpy()
    # Test text encoder
    assert np.allclose(mx_clip_tenc(tokens_mx), expected_last_hidden, atol=1e-5)


def test_vision_encoder():
    num_iters = 10
    b = 4
    h = 224
    w = 224
    c = 3

    mx_clip_venc = load_vision_encoder(TEST_CKPT)
    hf_clp_vision = CLIPVisionModel.from_pretrained(TEST_CKPT)

    for _ in range(num_iters):
        x = mx.random.normal((b, h, w, c))
        with torch.inference_mode():
            # Get expected
            x_tc = torch.tensor(x.tolist())
            x_tc = x_tc.permute((0, 3, 1, 2))
            expected_out = mx_clip_venc(x_tc)
            expected_last_hidden = expected_out.last_hidden_state.numpy()
            expected_pooler_output = expected_out.pooler_output.numpy()
        # Test vision encoder
        out = hf_clp_vision(x)
        assert np.allclose(out.last_hidden, expected_last_hidden, atol=1e-5)
        assert np.allclose(out.pooler_output, expected_pooler_output, atol=1e-5)


test_text_tokenizer()
test_text_encoder()
test_vision_encoder()
