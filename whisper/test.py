# Copyright © 2023 Apple Inc.

import json
import os
import subprocess
import unittest
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from convert import load_torch_model, quantize, torch_to_mlx
from mlx.utils import tree_flatten

import whisper
import whisper.audio as audio
import whisper.decoding as decoding
import whisper.load_models as load_models

MODEL_NAME = "tiny"
MLX_FP32_MODEL_PATH = "mlx_models/tiny_fp32"
MLX_FP16_MODEL_PATH = "mlx_models/tiny_fp16"
MLX_4BITS_MODEL_PATH = "mlx_models/tiny_quantized_4bits"
TEST_AUDIO = "whisper/assets/ls_test.flac"


def _save_model(save_dir, weights, config):
    mlx_path = Path(save_dir)
    mlx_path.mkdir(parents=True, exist_ok=True)

    # Save weights
    np.savez(str(mlx_path / "weights.npz"), **weights)

    # Save config.json with model_type
    with open(str(mlx_path / "config.json"), "w") as f:
        config["model_type"] = "whisper"
        json.dump(config, f, indent=4)

    config.pop("model_type", None)


def load_torch_and_mlx():
    torch_model = load_torch_model(MODEL_NAME)

    fp32_model = torch_to_mlx(torch_model, dtype=mx.float32)
    config = asdict(fp32_model.dims)
    weights = dict(tree_flatten(fp32_model.parameters()))
    _save_model(MLX_FP32_MODEL_PATH, weights, config)

    fp16_model = torch_to_mlx(torch_model, dtype=mx.float16)
    config = asdict(fp16_model.dims)
    weights = dict(tree_flatten(fp16_model.parameters()))
    _save_model(MLX_FP16_MODEL_PATH, weights, config)

    args = type("", (), {})()
    args.q_group_size = 64
    args.q_bits = 4
    weights, config = quantize(weights, config, args)
    _save_model(MLX_4BITS_MODEL_PATH, weights, config)

    return torch_model, fp32_model, fp16_model


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
        _, cls.model, _ = load_torch_and_mlx()
        data = audio.load_audio(TEST_AUDIO)
        data = audio.pad_or_trim(data)
        cls.mels = audio.log_mel_spectrogram(data)

    def test_torch_mlx(self):
        np.random.seed(10)

        torch_model = load_torch_model(MODEL_NAME)
        dims = torch_model.dims

        mels = np.random.randn(1, dims.n_mels, 3_000)
        tokens = np.random.randint(0, dims.n_vocab, (1, 20))

        torch_logits = forward_torch(torch_model, mels, tokens)

        mlx_logits = forward_mlx(self.model, mels, tokens)

        self.assertTrue(np.allclose(torch_logits, mlx_logits, atol=1e-2, rtol=1e-2))

    def test_fp16(self):
        mlx_model = load_models.load_model(MLX_FP16_MODEL_PATH, mx.float16)
        dims = mlx_model.dims
        mels = mx.array(np.random.randn(1, 3_000, dims.n_mels), mx.float16)
        tokens = mx.array(np.random.randint(0, dims.n_vocab, (1, 20)), mx.int32)
        logits = mlx_model(mels, tokens)
        self.assertEqual(logits.dtype, mx.float16)

    def test_quantized_4bits(self):
        mlx_model = load_models.load_model(MLX_4BITS_MODEL_PATH, mx.float16)
        dims = mlx_model.dims
        mels = mx.array(np.random.randn(1, 3_000, dims.n_mels), mx.float16)
        tokens = mx.array(np.random.randint(0, dims.n_vocab, (1, 20)), mx.int32)
        logits = mlx_model(mels, tokens)
        # Here, we just test if 4-bit models can forward, as the quantized tiny models struggle with accurate transcription
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
        result = whisper.transcribe(
            TEST_AUDIO, path_or_hf_repo=MLX_FP32_MODEL_PATH, fp16=False
        )
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

        result = whisper.transcribe(
            audio_file, path_or_hf_repo=MLX_FP32_MODEL_PATH, fp16=False
        )
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
                    self.assertAlmostEqual(seg[k], v, places=2)
                else:
                    self.assertEqual(seg[k], v)

        # Randomly check a couple of segments
        check_segment(result["segments"][5], expected_5)
        check_segment(result["segments"][73], expected_73)

    def test_transcribe_word_level_timestamps_confidence_scores(self):
        result = whisper.transcribe(
            TEST_AUDIO,
            path_or_hf_repo=MLX_FP16_MODEL_PATH,
            word_timestamps=True,
        )

        # result predicted with openai-whisper
        expected_0 = [
            {
                "word": " Then",
                "start": 0.0,
                "end": 0.94,
                "probability": 0.855542778968811,
            },
            {
                "word": " the",
                "start": 0.94,
                "end": 1.12,
                "probability": 0.6500106453895569,
            },
            {
                "word": " good",
                "start": 1.12,
                "end": 1.32,
                "probability": 0.5503873825073242,
            },
            {
                "word": " soul",
                "start": 1.32,
                "end": 1.56,
                "probability": 0.46757155656814575,
            },
            {
                "word": " openly",
                "start": 1.56,
                "end": 2.0,
                "probability": 0.9840946793556213,
            },
            {
                "word": " sorted",
                "start": 2.0,
                "end": 2.38,
                "probability": 0.24167272448539734,
            },
            {
                "word": " the",
                "start": 2.38,
                "end": 2.58,
                "probability": 0.9875414967536926,
            },
            {
                "word": " boat",
                "start": 2.58,
                "end": 2.8,
                "probability": 0.5856029391288757,
            },
            {
                "word": " and",
                "start": 2.8,
                "end": 2.98,
                "probability": 0.913351833820343,
            },
            {
                "word": " she",
                "start": 2.98,
                "end": 3.1,
                "probability": 0.9913808703422546,
            },
            {
                "word": " had",
                "start": 3.1,
                "end": 3.32,
                "probability": 0.9952940344810486,
            },
            {
                "word": " buoyed",
                "start": 3.32,
                "end": 3.58,
                "probability": 0.6411589980125427,
            },
            {
                "word": " so",
                "start": 3.58,
                "end": 3.8,
                "probability": 0.9682658314704895,
            },
            {
                "word": " long",
                "start": 3.8,
                "end": 4.06,
                "probability": 0.9953522682189941,
            },
            {
                "word": " in",
                "start": 4.06,
                "end": 4.26,
                "probability": 0.6745936870574951,
            },
            {
                "word": " secret",
                "start": 4.26,
                "end": 4.56,
                "probability": 0.9905064702033997,
            },
            {
                "word": " and",
                "start": 4.56,
                "end": 4.9,
                "probability": 0.856008768081665,
            },
            {
                "word": " bravely",
                "start": 4.9,
                "end": 5.28,
                "probability": 0.8477402329444885,
            },
        ]

        def check_words(words, expected_words):
            for word, expected_word in zip(words, expected_words):
                for k, v in expected_word.items():
                    if isinstance(v, float):
                        self.assertAlmostEqual(word[k], v, places=1)
                    else:
                        self.assertEqual(word[k], v)

        # Randomly check a couple of segments
        check_words(result["segments"][0]["words"], expected_0)


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
