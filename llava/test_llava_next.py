import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class Module:
    pass


sys.modules["mlx"] = _module("mlx")
sys.modules["mlx.core"] = _module("mlx.core", array=object)
sys.modules["mlx.nn"] = _module("mlx.nn", Module=Module)
sys.modules["huggingface_hub"] = _module(
    "huggingface_hub", snapshot_download=lambda *args, **kwargs: None
)
sys.modules["language"] = _module("language", LanguageModel=object, TextConfig=object)
sys.modules["vision"] = _module(
    "vision",
    VisionConfig=object,
    VisionModel=_module("VisionModel", sanitize=lambda weights: weights),
)

spec = importlib.util.spec_from_file_location(
    "llava_example", Path(__file__).with_name("llava.py")
)
llava = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llava)


def test_image_size_to_num_patches_includes_base_patch():
    num_patches = llava.image_size_to_num_patches(
        image_size=(48, 64),
        grid_pinpoints=[[336, 672], [672, 336], [672, 672]],
        patch_size=336,
    )

    assert num_patches == 3


def test_unpad_image_removes_resized_padding():
    tensor = np.ones((2, 4, 6))
    unpadded = llava.unpad_image(tensor, original_size=(2, 6))

    assert unpadded.shape == (2, 2, 6)
