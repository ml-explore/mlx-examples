# Copyright © 2024 Apple Inc.

"""
Voice Activity Detection (VAD) module for mlx-whisper.

Provides VAD preprocessing using a native MLX implementation of Silero VAD
to filter silent audio regions before transcription, improving speed and
reducing hallucinations on audio with significant silence.

This is a pure MLX implementation - no PyTorch required.
"""

import bisect
from dataclasses import dataclass
from typing import Dict, List, Optional

import mlx.core as mx
import numpy as np


def is_available() -> bool:
    """Check if VAD dependencies are available.

    Always returns True since MLX VAD has no external dependencies
    beyond MLX itself (which is already required for mlx-whisper).
    """
    return True


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
    """Native MLX wrapper for Silero VAD model.

    Loads the model lazily on first use to avoid import overhead
    when VAD is not needed.
    """

    def __init__(self, sample_rate: int = 16000):
        """Initialize VAD wrapper.

        Args:
            sample_rate: Audio sample rate (16000 or 8000)
        """
        self._model = None
        self._sample_rate = sample_rate

    def _load_model(self):
        """Load Silero VAD model (MLX native implementation)."""
        from .silero_vad import load_vad_model

        self._model = load_vad_model(sample_rate=self._sample_rate)

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
        if self._model is None:
            self._load_model()

        if options is None:
            options = VadOptions()

        return get_speech_timestamps(
            audio,
            self._model,
            threshold=options.threshold,
            min_speech_duration_ms=options.min_speech_duration_ms,
            max_speech_duration_s=options.max_speech_duration_s,
            min_silence_duration_ms=options.min_silence_duration_ms,
            speech_pad_ms=options.speech_pad_ms,
            sample_rate=self._sample_rate,
        )


def get_speech_timestamps(
    audio: np.ndarray,
    model,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = float("inf"),
    min_silence_duration_ms: int = 2000,
    speech_pad_ms: int = 400,
    sample_rate: int = 16000,
) -> List[Dict[str, int]]:
    """Detect speech timestamps in audio using Silero VAD.

    Args:
        audio: Audio waveform as numpy array
        model: SileroVADModel instance
        threshold: Speech detection threshold (0.0-1.0)
        min_speech_duration_ms: Minimum speech segment duration in ms
        max_speech_duration_s: Maximum speech segment duration in seconds
        min_silence_duration_ms: Minimum silence duration to split segments
        speech_pad_ms: Padding to add around speech segments in ms
        sample_rate: Audio sample rate

    Returns:
        List of dicts with 'start' and 'end' keys (sample indices)
    """
    # Model parameters
    window_size = model.window_size  # 512 for 16kHz, 256 for 8kHz

    # Convert time parameters to samples
    min_speech_samples = int(min_speech_duration_ms * sample_rate / 1000)
    min_silence_samples = int(min_silence_duration_ms * sample_rate / 1000)
    speech_pad_samples = int(speech_pad_ms * sample_rate / 1000)
    max_speech_samples = (
        int(max_speech_duration_s * sample_rate)
        if max_speech_duration_s < float("inf")
        else float("inf")
    )

    # Ensure audio is the right length (pad if needed)
    audio_length = len(audio)
    if audio_length % window_size != 0:
        pad_length = window_size - (audio_length % window_size)
        audio = np.pad(audio, (0, pad_length))

    # Process audio in chunks
    num_chunks = len(audio) // window_size
    probs = []

    h, c = model.reset_state(batch_size=1)

    for i in range(num_chunks):
        chunk = audio[i * window_size : (i + 1) * window_size]
        chunk_mx = mx.array(chunk.reshape(1, -1))

        prob, h, c = model(chunk_mx, h, c)
        mx.eval(prob, h, c)

        probs.append(float(prob[0, 0]))

    # Convert probabilities to speech segments
    speeches = []
    current_speech = None

    for i, prob in enumerate(probs):
        sample_pos = i * window_size

        if prob >= threshold:
            if current_speech is None:
                current_speech = {"start": sample_pos, "end": sample_pos + window_size}
            else:
                current_speech["end"] = sample_pos + window_size
        else:
            if current_speech is not None:
                # Check if silence is long enough to end segment
                silence_duration = sample_pos - current_speech["end"]
                if silence_duration >= min_silence_samples:
                    speeches.append(current_speech)
                    current_speech = None
                else:
                    # Continue current speech through short silence
                    current_speech["end"] = sample_pos + window_size

    # Don't forget the last segment
    if current_speech is not None:
        speeches.append(current_speech)

    # Filter by minimum duration
    speeches = [s for s in speeches if s["end"] - s["start"] >= min_speech_samples]

    # Split segments that exceed max duration
    if max_speech_samples < float("inf"):
        split_speeches = []
        for speech in speeches:
            duration = speech["end"] - speech["start"]
            if duration <= max_speech_samples:
                split_speeches.append(speech)
            else:
                # Split into smaller segments
                start = speech["start"]
                while start < speech["end"]:
                    end = min(start + max_speech_samples, speech["end"])
                    split_speeches.append({"start": start, "end": end})
                    start = end
        speeches = split_speeches

    # Apply padding
    for speech in speeches:
        speech["start"] = max(0, speech["start"] - speech_pad_samples)
        speech["end"] = min(audio_length, speech["end"] + speech_pad_samples)

    # Merge overlapping segments after padding
    if speeches:
        merged = [speeches[0]]
        for speech in speeches[1:]:
            if speech["start"] <= merged[-1]["end"]:
                merged[-1]["end"] = max(merged[-1]["end"], speech["end"])
            else:
                merged.append(speech)
        speeches = merged

    return speeches


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
