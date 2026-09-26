"""Parity test for the MLX beam search decoder against openai-whisper's.

Drives ``mlx_whisper.decoding.BeamSearchDecoder`` and openai-whisper's
torch ``BeamSearchDecoder`` side by side on identical synthetic logits and
checks that they produce the same tokens, completion flags, cumulative
log probabilities, KV-cache rearrange calls, and final candidates.

Requires ``mlx`` plus ``torch`` and ``openai-whisper`` for the reference
side (``pip install openai-whisper``). Skipped when openai-whisper is not
importable. Run from the whisper/ directory:

    python -m unittest discover tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import numpy as np
from mlx_whisper.decoding import BeamSearchDecoder as MlxBeamSearchDecoder

try:
    import torch
    from whisper.decoding import BeamSearchDecoder as TorchBeamSearchDecoder

    HAS_OPENAI_WHISPER = True
    OPENAI_WHISPER_SKIP_REASON = ""
except ImportError as e:
    HAS_OPENAI_WHISPER = False
    OPENAI_WHISPER_SKIP_REASON = (
        f"openai-whisper is not importable ({e}); install it with "
        "`pip install openai-whisper` and run the tests from the "
        "whisper/ directory"
    )

EOT = 5
VOCAB = 64
PREFIX_LEN = 3


class RecordingInference:
    """Duck-typed stand-in for Inference that records rearrange calls."""

    def __init__(self):
        self.rearrange_calls = []

    def rearrange_kv_cache(self, source_indices):
        self.rearrange_calls.append(list(source_indices))


def strip_trailing_eot(tokens):
    tokens = list(tokens)
    while tokens and tokens[-1] == EOT:
        tokens.pop()
    return tuple(tokens)


def sort_candidates(pairs):
    return sorted(pairs, key=lambda pair: (-pair[1], pair[0]))


def normalize_mlx(sequences, scores):
    sequences = np.array(sequences)
    scores = np.array(scores)
    return [
        sort_candidates(
            (strip_trailing_eot(row), float(score))
            for row, score in zip(sequences[i], scores[i])
        )
        for i in range(sequences.shape[0])
    ]


def normalize_torch(sequences, scores):
    return [
        sort_candidates(
            (strip_trailing_eot(seq.tolist()), float(score))
            for seq, score in zip(audio_sequences, audio_scores)
        )
        for audio_sequences, audio_scores in zip(sequences, scores)
    ]


def run_pair(beam_size, n_audio, patience, eot_boost_steps, n_steps, seed):
    """Run both decoders on the same logits stream and collect results."""
    n_batch = n_audio * beam_size
    rng = np.random.default_rng(seed)

    prefix = rng.integers(0, VOCAB, size=(n_batch, PREFIX_LEN), dtype=np.int64)
    prefix[prefix == EOT] = EOT + 1

    mlx_inference = RecordingInference()
    mlx_decoder = MlxBeamSearchDecoder(beam_size, EOT, mlx_inference, patience)
    mlx_tokens = mx.array(prefix)
    mlx_sum_logprobs = mx.zeros((n_batch,), mx.float32)

    torch_inference = RecordingInference()
    torch_decoder = TorchBeamSearchDecoder(beam_size, EOT, torch_inference, patience)
    torch_tokens = torch.tensor(prefix)
    torch_sum_logprobs = torch.zeros(n_batch)

    steps = []
    for step in range(n_steps):
        logits = rng.standard_normal((n_batch, VOCAB), dtype=np.float32)
        if step in eot_boost_steps:
            logits[:, EOT] += 6.0

        mlx_tokens, mlx_completed, mlx_sum_logprobs = mlx_decoder.update(
            mlx_tokens, mx.array(logits), mlx_sum_logprobs
        )
        torch_tokens, torch_completed = torch_decoder.update(
            torch_tokens, torch.from_numpy(logits), torch_sum_logprobs
        )

        # torch's update mutates sum_logprobs in place, so clone both
        # tensors to snapshot the values at this step.
        steps.append(
            {
                "mlx_tokens": np.array(mlx_tokens),
                "torch_tokens": torch_tokens.clone().numpy(),
                "mlx_completed": bool(mlx_completed),
                "torch_completed": bool(torch_completed),
                "mlx_sum_logprobs": np.array(mlx_sum_logprobs),
                "torch_sum_logprobs": torch_sum_logprobs.clone().numpy(),
            }
        )
        if steps[-1]["mlx_completed"]:
            break

    mlx_sequences, mlx_scores = mlx_decoder.finalize(
        mx.array(steps[-1]["mlx_tokens"].reshape(n_audio, beam_size, -1)),
        mx.array(steps[-1]["mlx_sum_logprobs"].reshape(n_audio, beam_size)),
    )
    torch_sequences, torch_scores = torch_decoder.finalize(
        torch.tensor(steps[-1]["torch_tokens"].reshape(n_audio, beam_size, -1)),
        torch.tensor(steps[-1]["torch_sum_logprobs"].reshape(n_audio, beam_size)),
    )

    return {
        "steps": steps,
        "mlx_rearrange_calls": mlx_inference.rearrange_calls,
        "torch_rearrange_calls": torch_inference.rearrange_calls,
        "mlx_final": normalize_mlx(mlx_sequences, mlx_scores),
        "torch_final": normalize_torch(torch_sequences, torch_scores),
    }


@unittest.skipUnless(HAS_OPENAI_WHISPER, OPENAI_WHISPER_SKIP_REASON)
class TestBeamSearchParity(unittest.TestCase):
    # Scenarios cover: beam sizes 1-3, multiple audio items, patience,
    # early termination via boosted EOT, and running out of steps so that
    # finalize has to drain unfinished beams.
    scenarios = [
        {
            "beam_size": 1,
            "n_audio": 1,
            "patience": None,
            "eot_boost_steps": {3},
            "n_steps": 8,
            "seed": 10,
        },
        {
            "beam_size": 2,
            "n_audio": 2,
            "patience": None,
            "eot_boost_steps": {2, 5},
            "n_steps": 10,
            "seed": 11,
        },
        {
            "beam_size": 3,
            "n_audio": 2,
            "patience": None,
            "eot_boost_steps": {3, 4, 6},
            "n_steps": 12,
            "seed": 12,
        },
        {
            "beam_size": 3,
            "n_audio": 1,
            "patience": 1.5,
            "eot_boost_steps": {2, 3, 4, 5, 6},
            "n_steps": 14,
            "seed": 13,
        },
        {
            "beam_size": 2,
            "n_audio": 2,
            "patience": None,
            "eot_boost_steps": set(),
            "n_steps": 6,
            "seed": 14,
        },
    ]

    def test_beam_search_parity(self):
        for scenario in self.scenarios:
            with self.subTest(**scenario):
                result = run_pair(**scenario)

                for step, record in enumerate(result["steps"]):
                    with self.subTest(step=step):
                        self.assertEqual(
                            record["mlx_tokens"].shape,
                            record["torch_tokens"].shape,
                        )
                        self.assertEqual(
                            record["mlx_tokens"].tolist(),
                            record["torch_tokens"].tolist(),
                        )
                        self.assertEqual(
                            record["mlx_completed"],
                            record["torch_completed"],
                        )
                        np.testing.assert_allclose(
                            record["mlx_sum_logprobs"],
                            record["torch_sum_logprobs"],
                            atol=1e-5,
                            rtol=1e-4,
                        )

                self.assertEqual(
                    result["mlx_rearrange_calls"],
                    result["torch_rearrange_calls"],
                )

                self.assertEqual(len(result["mlx_final"]), len(result["torch_final"]))
                for audio, (mlx_audio, torch_audio) in enumerate(
                    zip(result["mlx_final"], result["torch_final"])
                ):
                    with self.subTest(finalize_audio=audio):
                        self.assertEqual(len(mlx_audio), len(torch_audio))
                        for (mlx_seq, mlx_score), (torch_seq, torch_score) in zip(
                            mlx_audio, torch_audio
                        ):
                            self.assertEqual(mlx_seq, torch_seq)
                            self.assertAlmostEqual(mlx_score, torch_score, places=5)


if __name__ == "__main__":
    unittest.main()
