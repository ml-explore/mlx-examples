# Copyright © 2024 Apple Inc.

"""
Speaker diarization module for mlx-whisper.

Provides optional speaker diarization using pyannote.audio to identify
who is speaking when in multi-speaker audio.
"""

from typing import Dict, List, Optional, Union

import numpy as np

# Graceful dependency handling
_PYANNOTE_AVAILABLE = False
_PANDAS_AVAILABLE = False

try:
    import pandas as pd

    _PANDAS_AVAILABLE = True
except ImportError:
    pd = None

try:
    from pyannote.audio import Pipeline

    _PYANNOTE_AVAILABLE = True
except ImportError:
    Pipeline = None


def _extract_annotation(diarization_output):
    """Normalize diarization output to a pyannote Annotation-like object.

    Supports:
    - Legacy `Annotation` outputs with ``itertracks``
    - `DiarizeOutput` (pyannote.audio>=3.1) providing `speaker_diarization`
    - Dict-like outputs with `speaker_diarization` or `annotation`
    Falls back to `exclusive_speaker_diarization` when the primary annotation is
    present but empty.
    """

    if hasattr(diarization_output, "itertracks"):
        return diarization_output

    # pyannote.audio>=3.1 returns DiarizeOutput
    if hasattr(diarization_output, "speaker_diarization"):
        ann = diarization_output.speaker_diarization
        try:
            if hasattr(diarization_output, "exclusive_speaker_diarization") and len(ann) == 0:  # type: ignore[arg-type]
                ann = diarization_output.exclusive_speaker_diarization
        except Exception:
            pass
        return ann

    # Some variants may expose `.annotation`
    if hasattr(diarization_output, "annotation"):
        return diarization_output.annotation

    # dict-like outputs
    if isinstance(diarization_output, dict):
        if "speaker_diarization" in diarization_output:
            ann = diarization_output["speaker_diarization"]
            if (
                "exclusive_speaker_diarization" in diarization_output
                and hasattr(ann, "__len__")
                and len(ann) == 0
            ):
                ann = diarization_output["exclusive_speaker_diarization"]
            return ann
        if "annotation" in diarization_output:
            return diarization_output["annotation"]

    raise AttributeError(
        "Unsupported diarization output: expected an object with `itertracks` "
        "or a `speaker_diarization`/`annotation` attribute."
    )


def is_available() -> bool:
    """Check if diarization dependencies are available."""
    return _PYANNOTE_AVAILABLE and _PANDAS_AVAILABLE


class DiarizationUnavailableError(ImportError):
    """Raised when diarization is requested but dependencies are missing."""

    def __init__(self):
        missing = []
        if not _PYANNOTE_AVAILABLE:
            missing.append("pyannote.audio>=3.1")
        if not _PANDAS_AVAILABLE:
            missing.append("pandas")

        deps = ", ".join(missing)
        super().__init__(
            f"Diarization requires: {deps}\n"
            f"Install with: pip install {' '.join(missing)}\n"
            "Note: pyannote.audio requires a HuggingFace token for model access.\n"
            "See: https://huggingface.co/pyannote/speaker-diarization-3.1"
        )


class DiarizationPipeline:
    """Wrapper for pyannote.audio speaker diarization pipeline.

    Identifies speaker segments in audio, producing a timeline of
    who is speaking when.
    """

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-3.1",
        token: Optional[str] = None,
        device: str = "cpu",
    ):
        """Initialize diarization pipeline.

        Args:
            model_name: HuggingFace model ID for diarization
            use_auth_token: HuggingFace token (required for gated models)
            device: Device to run on ('cpu', 'cuda', 'mps')
        """
        if not is_available():
            raise DiarizationUnavailableError()

        import torch, pyannote
        from pyannote.audio import Pipeline

        torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
        torch.serialization.add_safe_globals([pyannote.audio.core.task.Specifications])
        torch.serialization.add_safe_globals([pyannote.audio.core.task.Problem])
        torch.serialization.add_safe_globals([pyannote.audio.core.task.Resolution])

        self.device = torch.device(device)
        self.model: Pipeline = Pipeline.from_pretrained(
            model_name, token=token
        )
        self.model.to(self.device)

    def __call__(
        self,
        audio: Union[str, np.ndarray],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        sample_rate: int = 16000,
    ):
        """Run speaker diarization on audio.

        Args:
            audio: Audio file path or waveform array
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            sample_rate: Sample rate of audio array

        Returns:
            pandas DataFrame with columns: start, end, speaker
        """
        import pandas as pd
        import torch

        if isinstance(audio, str):
            from .audio import load_audio

            audio = np.array(load_audio(audio))

        audio_data = {
            "waveform": torch.from_numpy(audio[None, :]).float(),
            "sample_rate": sample_rate,
        }

        diarization = self.model(
            audio_data,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        # Normalize output across pyannote versions
        annotation = _extract_annotation(diarization)

        # Convert to DataFrame
        segments = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})

        return pd.DataFrame(segments)


def assign_word_speakers(
    diarize_df, segments: List[Dict], fill_nearest: bool = False  # pandas DataFrame
) -> List[Dict]:
    """Assign speaker labels to transcript segments and words.

    Uses intersection-based assignment: each segment/word is assigned
    to the speaker with maximum time overlap.

    Args:
        diarize_df: DataFrame with start, end, speaker columns
        segments: List of transcript segments from transcribe()
        fill_nearest: If True, assign speaker even when no overlap

    Returns:
        Segments with 'speaker' field added
    """
    import numpy as np

    for seg in segments:
        # Calculate intersection with each diarization segment
        df_copy = diarize_df.copy()
        df_copy["intersection"] = np.minimum(
            df_copy["end"].values, seg["end"]
        ) - np.maximum(df_copy["start"].values, seg["start"])

        # Filter to overlapping segments
        if fill_nearest:
            dia_tmp = df_copy
        else:
            dia_tmp = df_copy[df_copy["intersection"] > 0]

        if len(dia_tmp) > 0:
            # Assign speaker with maximum intersection
            speaker = (
                dia_tmp.groupby("speaker")["intersection"]
                .sum()
                .sort_values(ascending=False)
                .index[0]
            )
            seg["speaker"] = speaker

        # Assign speakers to individual words
        if "words" in seg:
            for word in seg["words"]:
                if "start" in word and "end" in word:
                    df_copy["word_intersection"] = np.minimum(
                        df_copy["end"].values, word["end"]
                    ) - np.maximum(df_copy["start"].values, word["start"])

                    if fill_nearest:
                        word_dia = df_copy
                    else:
                        word_dia = df_copy[df_copy["word_intersection"] > 0]

                    if len(word_dia) > 0:
                        word_speaker = (
                            word_dia.groupby("speaker")["word_intersection"]
                            .sum()
                            .sort_values(ascending=False)
                            .index[0]
                        )
                        word["speaker"] = word_speaker

    return segments
