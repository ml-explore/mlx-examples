# Copyright © 2023-2024 Apple Inc.

import unittest
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mlx_whisper.decoding import (
    BeamSearchDecoder,
    DecodingOptions,
    DecodingTask,
    GreedyDecoder,
)


class FakeInference:
    def __init__(self):
        self.rearrange_calls = []

    def rearrange_kv_cache(self, source_indices):
        self.rearrange_calls.append(list(source_indices))


def logprobs(logits):
    logits = np.array(logits, dtype=np.float32)
    return logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))


class TestBeamSearchDecoder(unittest.TestCase):
    def test_basic_beam_expansion(self):
        inference = FakeInference()
        decoder = BeamSearchDecoder(
            beam_size=2, eot=9, inference=inference, patience=2.0
        )
        tokens = mx.array([[0], [0]], dtype=mx.int32)
        sum_logprobs = mx.array([0.0, -3.0], dtype=mx.float32)
        logits = mx.array(
            [
                [-10.0, 10.0, 9.0, 8.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
                [-10.0, 7.0, 6.0, 5.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
            ],
            dtype=mx.float32,
        )

        next_tokens, completed, next_logprobs = decoder.update(
            tokens, logits, sum_logprobs
        )

        self.assertFalse(completed)
        self.assertEqual(next_tokens.tolist(), [[0, 1], [0, 2]])
        self.assertEqual(inference.rearrange_calls, [[0, 0]])
        expected = logprobs(logits.tolist())[0, [1, 2]].tolist()
        self.assertTrue(np.allclose(next_logprobs.tolist(), expected, atol=1e-6))

    def test_eot_candidates_are_finished_not_active(self):
        inference = FakeInference()
        decoder = BeamSearchDecoder(
            beam_size=2, eot=9, inference=inference, patience=2.0
        )
        tokens = mx.array([[0, 1], [0, 2]], dtype=mx.int32)
        sum_logprobs = mx.array([0.0, -0.2], dtype=mx.float32)
        logits = mx.array(
            [
                [-10.0, -10.0, -10.0, 8.0, -10.0, -10.0, -10.0, -10.0, 7.0, 10.0],
                [-10.0, -10.0, -10.0, -10.0, 8.0, -10.0, -10.0, -10.0, 7.0, 9.0],
            ],
            dtype=mx.float32,
        )

        next_tokens, completed, _ = decoder.update(tokens, logits, sum_logprobs)

        self.assertFalse(completed)
        self.assertEqual(next_tokens.tolist(), [[0, 2, 4], [0, 1, 3]])
        self.assertTrue(all(row[-1] != 9 for row in next_tokens.tolist()))
        self.assertEqual(len(decoder.finished_sequences[0]), 2)
        self.assertTrue(all(seq[-1] == 9 for seq in decoder.finished_sequences[0]))

    def test_finalize_adds_eot_terminated_unfinished_beams(self):
        decoder = BeamSearchDecoder(beam_size=2, eot=9, inference=FakeInference())
        decoder.finished_sequences = [{}]
        tokens = mx.array([[[0, 3], [0, 4]]], dtype=mx.int32)
        sum_logprobs = mx.array([[-2.0, -1.0]], dtype=mx.float32)

        final_tokens, final_logprobs = decoder.finalize(tokens, sum_logprobs)

        self.assertEqual(final_tokens.tolist(), [[[0, 4, 9], [0, 3, 9]]])
        self.assertEqual(final_logprobs.tolist(), [[-1.0, -2.0]])

    def test_patience_controls_completion(self):
        inference = FakeInference()
        decoder = BeamSearchDecoder(
            beam_size=2, eot=9, inference=inference, patience=2.0
        )
        tokens = mx.array([[0, 1], [0, 2]], dtype=mx.int32)
        sum_logprobs = mx.array([0.0, -0.1], dtype=mx.float32)
        logits = mx.array(
            [
                [-10.0, 8.0, 7.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, 10.0],
                [-10.0, 7.0, 8.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, 10.0],
            ],
            dtype=mx.float32,
        )

        tokens, completed, sum_logprobs = decoder.update(tokens, logits, sum_logprobs)
        self.assertFalse(completed)
        self.assertEqual(len(decoder.finished_sequences[0]), 2)

        _, completed, _ = decoder.update(tokens, logits, sum_logprobs)
        self.assertTrue(completed)
        self.assertEqual(len(decoder.finished_sequences[0]), 4)

    def test_kv_cache_reorder_matches_returned_parent_order(self):
        inference = FakeInference()
        decoder = BeamSearchDecoder(beam_size=3, eot=9, inference=inference)
        tokens = mx.array([[0], [0], [5]], dtype=mx.int32)
        sum_logprobs = mx.array([0.0, -5.0, -1.0], dtype=mx.float32)
        logits = mx.array(
            [
                [-10.0, 10.0, 9.0, 8.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
                [-10.0, 8.0, 7.0, 6.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
                [-10.0, 9.0, 8.0, 7.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
            ],
            dtype=mx.float32,
        )

        next_tokens, _, _ = decoder.update(tokens, logits, sum_logprobs)

        self.assertEqual(next_tokens.tolist(), [[0, 1], [0, 2], [5, 1]])
        self.assertEqual(inference.rearrange_calls, [[0, 0, 2]])

    def test_batch_groups_do_not_mix_audio_items(self):
        inference = FakeInference()
        decoder = BeamSearchDecoder(beam_size=2, eot=9, inference=inference)
        tokens = mx.array([[10], [10], [20], [20]], dtype=mx.int32)
        sum_logprobs = mx.array([0.0, -1.0, 0.0, -1.0], dtype=mx.float32)
        logits = mx.array(
            [
                [-10.0, 10.0, 9.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
                [-10.0, 8.0, 7.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
                [-10.0, -10.0, -10.0, 10.0, 9.0, -10.0, -10.0, -10.0, -10.0, -10.0],
                [-10.0, -10.0, -10.0, 8.0, 7.0, -10.0, -10.0, -10.0, -10.0, -10.0],
            ],
            dtype=mx.float32,
        )

        next_tokens, _, _ = decoder.update(tokens, logits, sum_logprobs)

        self.assertEqual(next_tokens.tolist(), [[10, 1], [10, 2], [20, 3], [20, 4]])
        self.assertEqual(inference.rearrange_calls, [[0, 0, 2, 2]])


class TestBeamSearchOptions(unittest.TestCase):
    def test_option_validation_and_beam_decoder_construction(self):
        model = SimpleNamespace(
            is_multilingual=False,
            num_languages=0,
            dims=SimpleNamespace(n_text_ctx=16, n_vocab=51864),
        )

        task = DecodingTask(
            model,
            DecodingOptions(
                beam_size=2,
                language="en",
                suppress_tokens="",
                suppress_blank=False,
                without_timestamps=True,
            ),
        )
        self.assertIsInstance(task.decoder, BeamSearchDecoder)

        with self.assertRaises(ValueError):
            DecodingTask(model, DecodingOptions(beam_size=2, best_of=2))
        with self.assertRaises(ValueError):
            DecodingTask(model, DecodingOptions(patience=1.5))
        with self.assertRaises(ValueError):
            DecodingTask(model, DecodingOptions(length_penalty=1.1))

    def test_greedy_decoder_still_appends_argmax(self):
        decoder = GreedyDecoder(temperature=0.0, eot=9)
        tokens = mx.array([[0], [9]], dtype=mx.int32)
        logits = mx.array(
            [
                [-10.0, 3.0, 2.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
                [-10.0, 3.0, 2.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
            ],
            dtype=mx.float32,
        )
        sum_logprobs = mx.zeros(2)

        next_tokens, completed, _ = decoder.update(tokens, logits, sum_logprobs)

        self.assertEqual(next_tokens.tolist(), [[0, 1], [9, 9]])
        self.assertFalse(completed)


if __name__ == "__main__":
    unittest.main()
