"""Check the cross-attention KV skip in ``Inference.rearrange_kv_cache``.

``Inference.rearrange_kv_cache`` gathers only the self-attention cache and
leaves the cross-attention K/V untouched, relying on every beam of an
audio item sharing identical cross-attention rows. These tests check that
against the full gather in ``kv_experiments.rearrange_full_kv``.

Requires ``mlx`` only. Run from the whisper/ directory:

    python -m unittest discover tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
from mlx.utils import tree_map
from mlx_whisper.decoding import Inference


def rearrange_full_kv(kv_cache, source_indices):
    """Gather both the self- and cross-attention KV caches.

    Reference version of ``Inference.rearrange_kv_cache`` that also
    gathers the cross-attention K/V.

    Args:
        kv_cache: List over decoder blocks of ``(self_kv, cross_kv)``
            trees, as returned by ``AudioDecoder.__call__``.
        source_indices: Beam indices to gather along the batch dimension.
    """
    return [
        (
            tree_map(lambda x: x[source_indices], self_kv),
            tree_map(lambda x: x[source_indices], cross_kv),
        )
        for self_kv, cross_kv in kv_cache
    ]


def make_kv_cache(n_layers, n_audio, beam_size, seq_len, n_audio_ctx, n_state, seed):
    """Synthetic decoder KV cache with the production structure.

    Self-attention K/V rows differ per beam. Cross-attention K/V rows are
    identical across the beams of an audio item, as in ``DecodingTask.run``
    where the audio features are repeated for every beam.
    """
    rng = np.random.default_rng(seed)
    n_batch = n_audio * beam_size
    cache = []
    for _ in range(n_layers):
        k = mx.array(rng.standard_normal((n_batch, seq_len, n_state), dtype=np.float32))
        v = mx.array(rng.standard_normal((n_batch, seq_len, n_state), dtype=np.float32))
        cross_k = mx.array(
            np.repeat(
                rng.standard_normal((n_audio, n_audio_ctx, n_state), dtype=np.float32),
                beam_size,
                axis=0,
            )
        )
        cross_v = mx.array(
            np.repeat(
                rng.standard_normal((n_audio, n_audio_ctx, n_state), dtype=np.float32),
                beam_size,
                axis=0,
            )
        )
        cache.append(((k, v), (cross_k, cross_v)))
    return cache


def beam_local_permutation(n_audio, beam_size, seed):
    """Source indices that only permute beams within each audio item."""
    rng = np.random.default_rng(seed)
    indices = []
    for i in range(n_audio):
        group = np.arange(i * beam_size, (i + 1) * beam_size)
        indices.extend(rng.permutation(group).tolist())
    return indices


class TestCrossKVSkip(unittest.TestCase):
    def assert_bit_identical(self, cache_a, cache_b):
        self.assertEqual(len(cache_a), len(cache_b))
        for (self_a, cross_a), (self_b, cross_b) in zip(cache_a, cache_b):
            for a, b in zip(self_a, self_b):
                self.assertEqual(a.shape, b.shape)
                self.assertTrue(bool(mx.array_equal(a, b)))
            for a, b in zip(cross_a, cross_b):
                self.assertEqual(a.shape, b.shape)
                self.assertTrue(bool(mx.array_equal(a, b)))

    def test_identity_indices_are_noop(self):
        inference = Inference(model=None)
        inference.kv_cache = make_kv_cache(
            n_layers=2,
            n_audio=2,
            beam_size=3,
            seq_len=5,
            n_audio_ctx=8,
            n_state=4,
            seed=0,
        )
        before = inference.kv_cache
        inference.rearrange_kv_cache(list(range(6)))
        self.assertIs(inference.kv_cache, before)

    def test_skip_matches_full_gather(self):
        for seed in range(3):
            with self.subTest(seed=seed):
                cache = make_kv_cache(
                    n_layers=4,
                    n_audio=3,
                    beam_size=5,
                    seq_len=7,
                    n_audio_ctx=16,
                    n_state=8,
                    seed=seed,
                )
                inference = Inference(model=None)
                inference.kv_cache = cache
                indices = beam_local_permutation(3, 5, seed=seed)

                inference.rearrange_kv_cache(indices)
                expected = rearrange_full_kv(cache, indices)

                self.assert_bit_identical(inference.kv_cache, expected)
                # The self-attention rows must actually be permuted,
                # otherwise the comparison above is vacuous.
                np.testing.assert_array_equal(
                    np.array(inference.kv_cache[0][0][0]),
                    np.array(cache[0][0][0])[indices],
                )

    def test_cross_kv_is_not_copied(self):
        cache = make_kv_cache(
            n_layers=3,
            n_audio=2,
            beam_size=4,
            seq_len=6,
            n_audio_ctx=12,
            n_state=4,
            seed=1,
        )
        inference = Inference(model=None)
        inference.kv_cache = cache
        inference.rearrange_kv_cache(beam_local_permutation(2, 4, seed=1))
        for original, updated in zip(cache, inference.kv_cache):
            self.assertIs(updated[1], original[1])


if __name__ == "__main__":
    unittest.main()
