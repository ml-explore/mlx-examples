# Copyright © 2024 Apple Inc.

"""
Voice Activity Detection (VAD) module for mlx-whisper.

Provides optional VAD preprocessing using Silero VAD to filter
silent audio regions before transcription, improving speed and
reducing hallucinations on audio with significant silence.
"""

import bisect
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# Graceful dependency handling
_TORCH_AVAILABLE = False
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    pass


def is_available() -> bool:
    """Check if VAD dependencies are available."""
    return _TORCH_AVAILABLE


class VadUnavailableError(ImportError):
    """Raised when VAD is requested but dependencies are missing."""

    def __init__(self):
        super().__init__(
            "VAD requires PyTorch. Install with: pip install torch\n"
            "Or install mlx-whisper with VAD support: pip install mlx-whisper[vad]"
        )


@dataclass
class VadOptions:
    """Configuration options for Voice Activity Detection.

    Attributes:
        threshold: Speech detection threshold (0.0-1.0). Higher values
            require more confidence for speech detection. Default: 0.5
        min_speech_duration_ms: Minimum duration of speech segment in
            milliseconds. Shorter segments are discarded. Default: 250
        max_speech_duration_s: Maximum duration of speech segment in
            seconds. Longer segments are split. Default: inf
        min_silence_duration_ms: Minimum silence duration to split
            speech segments in milliseconds. Default: 2000
        speech_pad_ms: Padding added to each side of speech segments
            in milliseconds. Default: 400
    """

    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    max_speech_duration_s: float = float("inf")
    min_silence_duration_ms: int = 2000
    speech_pad_ms: int = 400


class SileroVAD:
    """Wrapper for Silero VAD model.

    Loads the model lazily on first use to avoid import overhead
    when VAD is not needed.
    """

    def __init__(self):
        self._model = None
        self._get_speech_timestamps = None

    def _load_model(self):
        """Load Silero VAD model via torch.hub."""
        if not is_available():
            raise VadUnavailableError()

        import torch

        self._model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        self._get_speech_timestamps = utils[0]

    def __call__(
        self, audio: np.ndarray, options: Optional[VadOptions] = None
    ) -> List[Dict[str, int]]:
        """Detect speech segments in audio.

        Args:
            audio: Audio waveform as numpy array (16kHz, mono)
            options: VAD configuration options

        Returns:
            List of dictionaries with 'start' and 'end' keys
            representing speech segment boundaries in samples.
        """
        import torch

        if self._model is None:
            self._load_model()

        if options is None:
            options = VadOptions()

        wav = torch.from_numpy(audio).float()

        speech_timestamps = self._get_speech_timestamps(
            wav,
            self._model,
            threshold=options.threshold,
            min_speech_duration_ms=options.min_speech_duration_ms,
            max_speech_duration_s=options.max_speech_duration_s,
            min_silence_duration_ms=options.min_silence_duration_ms,
            speech_pad_ms=options.speech_pad_ms,
            return_seconds=False,  # Return in samples
        )

        return speech_timestamps


def get_speech_chunks(
    audio: np.ndarray, timestamps: List[Dict[str, int]]
) -> np.ndarray:
    """Concatenate speech segments from audio.

    Args:
        audio: Full audio waveform
        timestamps: List of speech segment boundaries from VAD

    Returns:
        Concatenated audio containing only speech segments.
        Returns original audio if no timestamps provided.
    """
    if not timestamps:
        return audio

    chunks = [audio[ts["start"] : ts["end"]] for ts in timestamps]
    return np.concatenate(chunks)


class SpeechTimestampsMap:
    """Maps timestamps from VAD-filtered audio back to original timeline.

    When VAD removes silent segments, timestamps in the transcription
    refer to the filtered audio. This class provides conversion back
    to the original audio timeline.
    """

    def __init__(self, chunks: List[Dict[str, int]], sampling_rate: int = 16000):
        """Initialize timestamp mapping.

        Args:
            chunks: List of speech segment boundaries from VAD
            sampling_rate: Audio sample rate (default: 16000)
        """
        self.sampling_rate = sampling_rate
        self.chunk_end_sample: List[int] = []
        self.total_silence_before: List[float] = []

        previous_end = 0
        silent_samples = 0

        for chunk in chunks:
            silent_samples += chunk["start"] - previous_end
            previous_end = chunk["end"]
            self.chunk_end_sample.append(chunk["end"] - silent_samples)
            self.total_silence_before.append(silent_samples / sampling_rate)

    def get_original_time(self, time: float) -> float:
        """Convert filtered audio time to original audio time.

        Args:
            time: Timestamp in filtered audio (seconds)

        Returns:
            Corresponding timestamp in original audio (seconds)
        """
        sample = int(time * self.sampling_rate)
        # Use bisect_left: find the chunk this sample falls within
        chunk_idx = bisect.bisect_left(self.chunk_end_sample, sample)

        if chunk_idx >= len(self.chunk_end_sample):
            chunk_idx = len(self.chunk_end_sample) - 1

        if chunk_idx < 0 or not self.chunk_end_sample:
            return time

        return round(self.total_silence_before[chunk_idx] + time, 3)
