# Copyright © 2023-2024 Apple Inc.

from . import audio, decoding, load_models
from ._version import __version__
from .transcribe import transcribe, transcribe_with_diarization

# Optional modules (may not be available if dependencies are missing or incompatible)
try:
    from . import vad
except (ImportError, AttributeError):
    vad = None

try:
    from . import diarize
except (ImportError, AttributeError):
    diarize = None
