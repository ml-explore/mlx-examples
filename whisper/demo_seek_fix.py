#!/usr/bin/env python3
"""Demo: transcribe short audio with whisper-base.en to test seek behavior."""

import os
import string
import urllib.request
import mlx_whisper

# Download test audio
audio = "LJ037-0171.wav"
if not os.path.exists(audio):
    urllib.request.urlretrieve(f"https://keithito.com/LJ-Speech-Dataset/{audio}", audio)

# Expected transcription
expected = "The examination and testimony of the experts enabled the commission to conclude that five shots may have been fired"

# Transcribe
result = mlx_whisper.transcribe(audio, path_or_hf_repo="mlx-community/whisper-base.en-mlx")

# Compute accuracy
strip = str.maketrans("", "", string.punctuation)
expected_words = set(expected.lower().translate(strip).split())
actual_words = set(result["text"].lower().translate(strip).split())
accuracy = len(expected_words & actual_words) / len(expected_words) * 100

# Output
print(f"Expected: {expected}")
print(f"Actual:   {result['text'].strip()}")
print(f"Accuracy: {accuracy:.0f}%")
print(f"Segments: {len(result['segments'])}")
for s in result["segments"]:
    print(f"  [{s['start']:.2f}s - {s['end']:.2f}s]{s['text']}")
