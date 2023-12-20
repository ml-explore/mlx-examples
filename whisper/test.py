# Copyright © 2023 Apple Inc.

import os
import subprocess
import unittest

import mlx.core as mx
import numpy as np
import torch

import whisper
import whisper.audio as audio
import whisper.decoding as decoding
import whisper.load_models as load_models
import whisper.torch_whisper as torch_whisper

TEST_AUDIO = "whisper/assets/ls_test.flac"


def forward_torch(model, mels, tokens):
    mels = torch.Tensor(mels).to(torch.float32)
    tokens = torch.Tensor(tokens).to(torch.int32)
    with torch.no_grad():
        logits = model.forward(mels, tokens)
    return logits.numpy()


def forward_mlx(model, mels, tokens):
    mels = mx.array(mels.transpose(0, 2, 1))
    tokens = mx.array(tokens, mx.int32)
    logits = model(mels, tokens)
    return np.array(logits)


class TestWhisper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_models.load_model("tiny", dtype=mx.float32)
        data = audio.load_audio(TEST_AUDIO)
        data = audio.pad_or_trim(data)
        cls.mels = audio.log_mel_spectrogram(data)

    def test_torch_mlx(self):
        np.random.seed(10)

        torch_model = load_models.load_torch_model("tiny")
        dims = torch_model.dims

        mels = np.random.randn(1, dims.n_mels, 3_000)
        tokens = np.random.randint(0, dims.n_vocab, (1, 20))

        torch_logits = forward_torch(torch_model, mels, tokens)

        mlx_model = load_models.torch_to_mlx(torch_model, mx.float32)
        mlx_logits = forward_mlx(mlx_model, mels, tokens)

        self.assertTrue(np.allclose(torch_logits, mlx_logits, atol=1e-2, rtol=1e-2))

    def test_fp16(self):
        mlx_model = load_models.load_model("tiny", dtype=mx.float16)
        dims = mlx_model.dims
        mels = mx.array(np.random.randn(1, 3_000, dims.n_mels), mx.float16)
        tokens = mx.array(np.random.randint(0, dims.n_vocab, (1, 20)), mx.int32)
        logits = mlx_model(mels, tokens)
        self.assertEqual(logits.dtype, mx.float16)

    def test_decode_lang(self):
        options = decoding.DecodingOptions(task="lang_id", fp16=False)
        result = decoding.decode(self.model, self.mels, options)
        self.assertEqual(result.language, "en")
        self.assertEqual(len(result.language_probs), 99)
        self.assertAlmostEqual(
            result.language_probs["en"], 0.9947282671928406, places=5
        )

    def test_decode_greedy(self):
        result = decoding.decode(self.model, self.mels, fp16=False)
        self.assertEqual(result.language, "en")
        self.assertEqual(
            result.tokens,
            [
                50364,
                1396,
                264,
                665,
                5133,
                23109,
                25462,
                264,
                6582,
                293,
                750,
                632,
                42841,
                292,
                370,
                938,
                294,
                4054,
                293,
                12653,
                356,
                50620,
                50620,
                23563,
                322,
                3312,
                13,
                50680,
            ],
        )
        self.assertEqual(
            result.text,
            (
                "Then the good soul openly sorted the boat and she "
                "had buoyed so long in secret and bravely stretched on alone."
            ),
        )
        self.assertAlmostEqual(result.avg_logprob, -0.4975455382774616, places=3)
        self.assertAlmostEqual(result.no_speech_prob, 0.009631240740418434, places=4)
        self.assertAlmostEqual(result.compression_ratio, 1.2359550561797752)

        # Small temp should give the same results
        result = decoding.decode(self.model, self.mels, temperature=1e-8, fp16=False)

        self.assertEqual(
            result.text,
            (
                "Then the good soul openly sorted the boat and she "
                "had buoyed so long in secret and bravely stretched on alone."
            ),
        )
        self.assertAlmostEqual(result.avg_logprob, -0.4975455382774616, places=3)
        self.assertAlmostEqual(result.no_speech_prob, 0.009631240740418434, places=4)
        self.assertAlmostEqual(result.compression_ratio, 1.2359550561797752)

    def test_transcribe(self):
        result = whisper.transcribe(TEST_AUDIO, fp16=False)
        self.assertEqual(
            result["text"],
            (
                " Then the good soul openly sorted the boat and she "
                "had buoyed so long in secret and bravely stretched on alone."
            ),
        )

    def test_transcribe_alice(self):
        audio_file = os.path.join(
            os.path.expanduser("~"),
            ".cache/whisper/alice.mp3",
        )
        if not os.path.exists(audio_file):
            print("To run this test download the alice in wonderland audiobook:")
            print("bash path_to_whisper_repo/whisper/assets/download_alice.sh")
            return

        result = whisper.transcribe(audio_file, fp16=False)
        self.assertEqual(len(result["text"]), 10920)
        self.assertEqual(result["language"], "en")
        self.assertEqual(len(result["segments"]), 77)

        expected_5 = {
            "id": 5,
            "seek": 2800,
            "start": 40.0,
            "end": 46.0,
            "text": " Oh my poor little feet, I wonder who will put on your shoes and stockings for you now tears.",
            "tokens": [
                50964,
                876,
                452,
                4716,
                707,
                3521,
                11,
                286,
                2441,
                567,
                486,
                829,
                322,
                428,
                6654,
                293,
                4127,
                1109,
                337,
                291,
                586,
                10462,
                13,
                51264,
            ],
            "temperature": 0.0,
            "avg_logprob": -0.19670599699020386,
            "compression_ratio": 1.5991379310344827,
            "no_speech_prob": 0.09746722131967545,
        }

        expected_73 = {
            "id": 73,
            "seek": 70700,
            "start": 707.0,
            "end": 715.0,
            "text": " let us get to the shore, and then I'll tell you my history, and you'll understand why it is that I hate cats and dogs.",
            "tokens": [
                50364,
                718,
                505,
                483,
                281,
                264,
                17805,
                11,
                293,
                550,
                286,
                603,
                980,
                291,
                452,
                2503,
                11,
                293,
                291,
                603,
                1223,
                983,
                309,
                307,
                300,
                286,
                4700,
                11111,
                293,
                7197,
                13,
                50764,
            ],
            "temperature": 0.0,
            "avg_logprob": -0.1350895343440594,
            "compression_ratio": 1.6208333333333333,
            "no_speech_prob": 0.002246702555567026,
        }

        def check_segment(seg, expected):
            for k, v in expected.items():
                if isinstance(v, float):
                    self.assertAlmostEqual(seg[k], v, places=3)
                else:
                    self.assertEqual(seg[k], v)

        # Randomly check a couple of segments
        check_segment(result["segments"][5], expected_5)
        check_segment(result["segments"][73], expected_73)


class TestAudio(unittest.TestCase):
    def test_load(self):
        data = audio.load_audio(TEST_AUDIO)
        data_8k = audio.load_audio(TEST_AUDIO, 8000)
        n = 106640
        self.assertTrue(data.shape, (n,))
        self.assertTrue(data.dtype, np.float32)
        self.assertTrue(data_8k.shape, (n // 2,))

    def test_pad(self):
        data = audio.load_audio(TEST_AUDIO)
        data = audio.pad_or_trim(data, 20_000)
        self.assertTrue(data.shape, [20_000])

    def test_mel_spec(self):
        mels = audio.log_mel_spectrogram(TEST_AUDIO)
        self.assertTrue(mels.shape, [80, 400])
        self.assertTrue(mels.dtype, mx.float32)


if __name__ == "__main__":
    unittest.main()
