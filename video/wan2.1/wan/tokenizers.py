# Copyright © 2026 Apple Inc.

"""
T5 tokenizer for Wan2.1.

Uses the `tokenizers` library (HuggingFace tokenizers) to load tokenizer.json,
avoiding PyTorch and sentencepiece dependencies.
"""

from typing import Any, Dict

import mlx.core as mx


class T5Tokenizer:
    """Pure tokenizer wrapper using HuggingFace tokenizers library."""

    def __init__(self, tokenizer_path: str):
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self.pad_token_id = self._tokenizer.token_to_id("<pad>") or 0

    def __call__(
        self,
        text: str,
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
    ) -> Dict[str, Any]:
        encoded = self._tokenizer.encode(text)
        input_ids = encoded.ids

        if truncation and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]

        attention_mask = [1] * len(input_ids)

        if padding == "max_length" and len(input_ids) < max_length:
            pad_length = max_length - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length

        return {
            "input_ids": mx.array([input_ids], dtype=mx.int32),
            "attention_mask": mx.array([attention_mask], dtype=mx.int32),
        }
