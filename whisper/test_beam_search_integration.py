# Copyright © 2023-2024 Apple Inc.

import os
import subprocess
import sys
import tempfile
import unittest

import mlx.core as mx

import mlx_whisper
import mlx_whisper.audio as audio
from mlx_whisper.load_models import load_model


RUN_INTEGRATION = os.environ.get("RUN_MLX_WHISPER_INTEGRATION") == "1"
MODEL_NAME = os.environ.get("MLX_WHISPER_TEST_MODEL", "mlx-community/whisper-tiny")
TEST_AUDIO = os.path.join(
    os.path.dirname(__file__), "mlx_whisper", "assets", "ls_test.flac"
)


@unittest.skipUnless(
    RUN_INTEGRATION,
    "set RUN_MLX_WHISPER_INTEGRATION=1 to run MLX Whisper integration tests",
)
class TestBeamSearchIntegration(unittest.TestCase):
    def test_transcribe_without_timestamps(self):
        result = mlx_whisper.transcribe(
            TEST_AUDIO,
            path_or_hf_repo=MODEL_NAME,
            beam_size=3,
            patience=None,
            length_penalty=None,
            temperature=0.0,
            without_timestamps=True,
            language="en",
        )

        self.assertIsInstance(result["text"], str)
        self.assertTrue(result["text"].strip())

    def test_transcribe_with_timestamps(self):
        result = mlx_whisper.transcribe(
            TEST_AUDIO,
            path_or_hf_repo=MODEL_NAME,
            beam_size=3,
            temperature=0.0,
            without_timestamps=False,
            language="en",
        )

        self.assertTrue(result["segments"])
        for segment in result["segments"]:
            self.assertLessEqual(segment["start"], segment["end"])
            self.assertIsInstance(segment["tokens"], list)

    def test_cli_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            command = [
                sys.executable,
                "-m",
                "mlx_whisper.cli",
                TEST_AUDIO,
                "--model",
                MODEL_NAME,
                "--beam-size",
                "3",
                "--temperature",
                "0",
                "--language",
                "en",
                "--output-dir",
                tmpdir,
                "--output-format",
                "txt",
                "--verbose",
                "False",
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=False)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(os.listdir(tmpdir))

    def test_batch_decode_smoke(self):
        model = load_model(MODEL_NAME, mx.float16)
        data = audio.pad_or_trim(audio.load_audio(TEST_AUDIO))
        mel = audio.log_mel_spectrogram(data)
        batch = mx.stack([mel, mel])

        results = model.decode(
            batch,
            beam_size=2,
            temperature=0.0,
            without_timestamps=True,
            language="en",
        )

        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.text for result in results))


if __name__ == "__main__":
    unittest.main()
