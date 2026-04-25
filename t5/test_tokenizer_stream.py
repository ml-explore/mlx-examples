"""Tests for streaming detokenization in t5.py.

Reproduces and guards against the bug reported in
https://github.com/ml-explore/mlx-examples/issues/1021 where streaming
generation emitted raw subword markers (e.g. ``Ġ``/``Ċ`` for the byte-level
BPE tokenizer used by ``Salesforce/codet5p-220m``) instead of plain text.

The previous implementation called ``convert_ids_to_tokens`` per generated
token and only stripped the SentencePiece ``▁`` marker, which is wrong for
byte-level BPE tokenizers and for SentencePiece tokens that use byte-fallback.
The fix decodes the running prefix with the HuggingFace tokenizer's own
``decode`` and yields the new substring on each step.
"""

from types import SimpleNamespace

import pytest

from t5 import Tokenizer


def _make_tokenizer(model_name: str, decoder_start_id: int = 0) -> Tokenizer:
    config = SimpleNamespace(decoder_start_token_id=decoder_start_id)
    return Tokenizer(config, model_name)


def _stream_decode(tokenizer: Tokenizer, ids):
    """Mimic the streaming loop in ``t5.py``'s ``__main__``."""
    out = []
    state = tokenizer.new_stream_state()
    for tid in ids:
        out.append(tokenizer.stream_decode(state, int(tid)))
    return "".join(out)


@pytest.mark.parametrize(
    "model_name,prompt",
    [
        ("t5-small", "translate English to German: That is good."),
        ("Salesforce/codet5p-220m", "def print_hello_world():"),
    ],
)
def test_stream_decode_matches_full_decode(model_name: str, prompt: str) -> None:
    tokenizer = _make_tokenizer(model_name)
    ids = tokenizer._tokenizer(prompt, return_tensors="np")["input_ids"][0].tolist()

    expected = tokenizer._tokenizer.decode(ids, skip_special_tokens=True)
    streamed = _stream_decode(tokenizer, ids)

    assert streamed == expected


def test_stream_decode_strips_byte_level_bpe_markers() -> None:
    """Regression: ``Ġ`` / ``Ċ`` must never leak into streamed output."""
    tokenizer = _make_tokenizer("Salesforce/codet5p-220m")
    ids = tokenizer._tokenizer('def hello():\n    print("hi")', return_tensors="np")[
        "input_ids"
    ][0].tolist()

    streamed = _stream_decode(tokenizer, ids)

    assert "Ġ" not in streamed
    assert "Ċ" not in streamed
