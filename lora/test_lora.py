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


sys.modules["mlx"] = _module("mlx")
sys.modules["mlx.core"] = _module("mlx.core")
sys.modules["mlx.nn"] = _module(
    "mlx.nn",
    losses=_module("mlx.nn.losses", cross_entropy=lambda logits, targets: logits),
    value_and_grad=lambda model, loss: None,
)
sys.modules["mlx.optimizers"] = _module("mlx.optimizers", Adam=object)
sys.modules["mlx.utils"] = _module("mlx.utils", tree_flatten=lambda tree: [])
sys.modules["models"] = _module("models", LoRALinear=object)
sys.modules["utils"] = _module("utils")

sys.path.insert(0, str(Path(__file__).parent))
spec = importlib.util.spec_from_file_location(
    "lora_example", Path(__file__).with_name("lora.py")
)
lora = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lora)


class DummyTokenizer:
    eos_token_id = 0
    add_eos_token = True

    def encode(self, text, add_special_tokens=True):
        tokens = [ord(c) for c in text]
        if add_special_tokens:
            tokens = [1] + tokens
            if self.add_eos_token:
                tokens.append(self.eos_token_id)
        return tokens


def test_encode_dataset_item_with_text_trains_on_all_tokens():
    tokens, loss_mask = lora.encode_dataset_item("abc", DummyTokenizer())

    assert tokens == [1, 97, 98, 99, 0]
    np.testing.assert_array_equal(loss_mask, np.ones(4, dtype=np.float32))


def test_encode_dataset_item_with_prompt_masks_prompt_tokens():
    tokens, loss_mask = lora.encode_dataset_item(
        {"prompt": "[INST] Say hi [/INST]", "text": "Hi."},
        DummyTokenizer(),
    )

    prompt_tokens = DummyTokenizer().encode("[INST] Say hi [/INST]")[:-1]
    assert tokens[: len(prompt_tokens)] == prompt_tokens
    assert tokens[-4:] == [72, 105, 46, 0]
    np.testing.assert_array_equal(
        loss_mask,
        np.array([0] * (len(prompt_tokens) - 1) + [1, 1, 1, 1], dtype=np.float32),
    )
