# Copyright © 2026 Apple Inc.

import unittest
from types import SimpleNamespace

import mlx.core as mx

from mlx_whisper.decoding import DecodingOptions, decode


class FakeDecoderModel:
    def __init__(self):
        self.dims = SimpleNamespace(
            n_audio_ctx=2,
            n_audio_state=3,
            n_text_ctx=8,
            n_vocab=51864,
        )
        self.is_multilingual = False
        self.num_languages = 0

    def decoder(self, tokens, audio_features, kv_cache=None):
        logits = mx.full(
            (tokens.shape[0], tokens.shape[1], self.dims.n_vocab),
            -10.0,
            dtype=mx.float32,
        )
        logits[:, -1, 100] = 10.0
        return logits, kv_cache, None


class TestReturnCandidates(unittest.TestCase):
    def test_decode_returns_ranked_candidates_when_requested(self):
        model = FakeDecoderModel()
        mel = mx.zeros((model.dims.n_audio_ctx, model.dims.n_audio_state))
        options = DecodingOptions(
            language="en",
            sample_len=1,
            suppress_blank=False,
            suppress_tokens="",
            without_timestamps=True,
            fp16=False,
            return_candidates=True,
        )

        result = decode(model, mel, options)

        self.assertEqual(result.tokens, [100])
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.candidates[0]["tokens"], [100])
        self.assertTrue(result.candidates[0]["selected"])


if __name__ == "__main__":
    unittest.main()
