# Copyright © 2024 Apple Inc.

import unittest
from pathlib import Path

from huggingface_hub import snapshot_download
from mlx_lm.tokenizer_utils import (
    BPEStreamingDetokenizer,
    NaiveStreamingDetokenizer,
    SPMStreamingDetokenizer,
    load_tokenizer,
)


class TestTokenizers(unittest.TestCase):

    def download_tokenizer(self, repo):
        path = Path(
            snapshot_download(
                repo_id=repo,
                allow_patterns=[
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "tokenizer.model",
                ],
            )
        )
        return load_tokenizer(path)

    def check_tokenizer(self, tokenizer):
        def check(tokens):
            expected_text = tokenizer.decode(tokens)
            detokenizer = tokenizer.detokenizer
            detokenizer.reset()
            text = ""
            for e, t in enumerate(tokens):
                detokenizer.add_token(t)
                seg = detokenizer.last_segment
                text += seg
                self.assertEqual(detokenizer.tokens, tokens[: e + 1])
            detokenizer.finalize()
            text += detokenizer.last_segment
            self.assertEqual(text, expected_text)

        tokens = tokenizer.encode("こんにちは！私の名前はAI")
        check(tokens)

        tokens = tokenizer.encode("a ,b")
        check(tokens)

        tokens = tokenizer.encode('{"why_its_funny" :"a_joke_explainer" ,"rating":3.5}')
        check(tokens)

        tokens = tokenizer.encode("3 3")
        check(tokens)

        tokens = tokenizer.encode("import 'package:flutter/material.dart';")
        check(tokens)

        tokens = tokenizer.encode("hello\nworld")
        check(tokens)

    def test_tokenizers(self):
        tokenizer_repos = [
            ("mlx-community/Qwen1.5-0.5B-Chat-4bit", BPEStreamingDetokenizer),
            ("mlx-community/Mistral-7B-v0.2-4bit", SPMStreamingDetokenizer),
            ("mlx-community/Phi-3.5-mini-instruct-4bit", SPMStreamingDetokenizer),
            ("mlx-community/Mistral-7B-Instruct-v0.3", SPMStreamingDetokenizer),
            ("mlx-community/Llama-3.2-1B-Instruct-4bit", BPEStreamingDetokenizer),
            ("mlx-community/Falcon3-7B-Instruct-4bit", BPEStreamingDetokenizer),
        ]
        for tokenizer_repo, expected_detokenizer in tokenizer_repos:
            with self.subTest(tokenizer=tokenizer_repo):
                tokenizer = self.download_tokenizer(tokenizer_repo)
                tokenizer.decode([0, 1, 2])
                self.assertTrue(isinstance(tokenizer.detokenizer, expected_detokenizer))
                self.check_tokenizer(tokenizer)

        # Try one with a naive detokenizer
        tokenizer = self.download_tokenizer("mlx-community/Llama-3.2-1B-Instruct-4bit")
        tokenizer._detokenizer = NaiveStreamingDetokenizer(tokenizer)
        self.check_tokenizer(tokenizer)

    def test_special_tokens(self):
        tokenizer_repo = "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx"
        tokenizer = self.download_tokenizer(tokenizer_repo)

        detokenizer = tokenizer.detokenizer
        detokenizer.reset()
        detokenizer.add_token(tokenizer.eos_token_id)
        detokenizer.finalize()

        self.assertEqual(detokenizer.last_segment, tokenizer.eos_token)


if __name__ == "__main__":
    unittest.main()
