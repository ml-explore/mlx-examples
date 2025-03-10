# Copyright © 2023-2024 Apple Inc.

import os

from ._version import __version__

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from .utils import convert, generate, load, stream_generate


def get_estimate_mem():
    from .estimate_memory import estimate_mem

    return estimate_mem
