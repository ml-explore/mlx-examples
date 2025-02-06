# Copyright © 2023-2024 Apple Inc.

import argparse
import copy
import hashlib
import json
import os
import urllib
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import List

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from mlx_whisper import torch_whisper
from mlx_whisper.whisper import ModelDimensions, Whisper
from tqdm import tqdm

_VALID_DTYPES = {"float16", "float32"}

_MODELS = {
    "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large-v1": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
    "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    "large-v3": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    "large": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    "large-v3-turbo": "https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt",
    "turbo": "https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt",
}

# base85-encoded (n_layers, n_heads) boolean arrays indicating the cross-attention heads that are
# highly correlated to the word-level timing, i.e. the alignment between audio and text tokens.
_ALIGNMENT_HEADS = {
    "tiny.en": b"ABzY8J1N>@0{>%R00Bk>$p{7v037`oCl~+#00",
    "tiny": b"ABzY8bu8Lr0{>%RKn9Fp%m@SkK7Kt=7ytkO",
    "base.en": b"ABzY8;40c<0{>%RzzG;p*o+Vo09|#PsxSZm00",
    "base": b"ABzY8KQ!870{>%RzyTQH3`Q^yNP!>##QT-<FaQ7m",
    "small.en": b"ABzY8>?_)10{>%RpeA61k&I|OI3I$65C{;;pbCHh0B{qLQ;+}v00",
    "small": b"ABzY8DmU6=0{>%Rpa?J`kvJ6qF(V^F86#Xh7JUGMK}P<N0000",
    "medium.en": b"ABzY8usPae0{>%R7<zz_OvQ{)4kMa0BMw6u5rT}kRKX;$NfYBv00*Hl@qhsU00",
    "medium": b"ABzY8B0Jh+0{>%R7}kK1fFL7w6%<-Pf*t^=N)Qr&0RR9",
    "large-v1": b"ABzY8r9j$a0{>%R7#4sLmoOs{s)o3~84-RPdcFk!JR<kSfC2yj",
    "large-v2": b"ABzY8zd+h!0{>%R7=D0pU<_bnWW*tkYAhobTNnu$jnkEkXqp)j;w1Tzk)UH3X%SZd&fFZ2fC2yj",
    "large-v3": b"ABzY8gWO1E0{>%R7(9S+Kn!D~%ngiGaR?*L!iJG9p-nab0JQ=-{D1-g00",
    "large": b"ABzY8gWO1E0{>%R7(9S+Kn!D~%ngiGaR?*L!iJG9p-nab0JQ=-{D1-g00",
    "large-v3-turbo": b"ABzY8j^C+e0{>%RARaKHP%t(lGR*)0g!tONPyhe`",
    "turbo": b"ABzY8j^C+e0{>%RARaKHP%t(lGR*)0g!tONPyhe`",
}


def _download(url: str, root: str) -> str:
    os.makedirs(root, exist_ok=True)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, os.path.basename(url))

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        with open(download_target, "rb") as f:
            model_bytes = f.read()
        if hashlib.sha256(model_bytes).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(
                f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
            )

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    with open(download_target, "rb") as fid:
        model_bytes = fid.read()
    if hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model."
        )

    return download_target


def available_models() -> List[str]:
    """Returns the names of available models"""
    return list(_MODELS.keys())


def hf_to_pt(weights, config):
    config = {
        "n_mels": config["num_mel_bins"],
        "n_audio_ctx": config["max_source_positions"],
        "n_audio_state": config["d_model"],
        "n_audio_head": config["encoder_attention_heads"],
        "n_audio_layer": config["encoder_layers"],
        "n_vocab": config["vocab_size"],
        "n_text_ctx": config["max_target_positions"],
        "n_text_state": config["d_model"],
        "n_text_head": config["decoder_attention_heads"],
        "n_text_layer": config["decoder_layers"],
    }

    def remap(k):
        k = k.replace("model.", "")
        k = k.replace(".layers", ".blocks")
        k = k.replace(".self_attn", ".attn")
        k = k.replace(".attn_layer_norm", ".attn_ln")
        k = k.replace(".encoder_attn.", ".cross_attn.")
        k = k.replace(".encoder_attn_layer_norm", ".cross_attn_ln")
        k = k.replace(".final_layer_norm", ".mlp_ln")
        k = k.replace(".q_proj", ".query")
        k = k.replace(".k_proj", ".key")
        k = k.replace(".v_proj", ".value")
        k = k.replace(".out_proj", ".out")
        k = k.replace(".fc1", ".mlp1")
        k = k.replace(".fc2", ".mlp2")
        k = k.replace("embed_positions.weight", "positional_embedding")
        k = k.replace("decoder.embed_tokens", "decoder.token_embedding")
        k = k.replace("encoder.layer_norm", "encoder.ln_post")
        k = k.replace("decoder.layer_norm", "decoder.ln")
        return k

    # token embeddings are shared with output projection
    weights.pop("proj_out.weight", None)
    weights = {remap(k): v for k, v in weights.items()}
    return weights, config


def load_torch_weights_and_config(
    name_or_path: str,
    download_root: str = None,
):
    if download_root is None:
        download_root = os.path.join(os.path.expanduser("~"), ".cache/whisper")

    # todo: accept alignment_heads of local Pytorch checkpoint
    alignment_heads = None
    if name_or_path in _MODELS:
        alignment_heads = _ALIGNMENT_HEADS[name_or_path]
        name_or_path = _download(_MODELS[name_or_path], download_root)
    elif not Path(name_or_path).exists():
        # Try downloading from HF
        from huggingface_hub import snapshot_download

        name_or_path = snapshot_download(
            repo_id=name_or_path,
            allow_patterns=[
                "*.json",
                "pytorch_model.bin",
                "model.safetensors",
                "*.txt",
            ],
        )

    if name_or_path.endswith(".pt"):
        checkpoint = torch.load(name_or_path, map_location="cpu", weights_only=False)
        weights, config = checkpoint["model_state_dict"], checkpoint["dims"]
    else:
        name_or_path = Path(name_or_path)
        pt_path = name_or_path / "pytorch_model.bin"
        if pt_path.is_file():
            weights = torch.load(pt_path, map_location="cpu")
        else:
            weights = mx.load(str(name_or_path / "model.safetensors"))
        with open(name_or_path / "config.json", "r") as fp:
            config = json.load(fp)
        weights, config = hf_to_pt(weights, config)

    return weights, config, alignment_heads


def load_torch_model(
    name_or_path: str,
    download_root: str = None,
) -> torch_whisper.Whisper:
    """
    Load a Whisper ASR model

    Parameters
    ----------
    name_or_path : str
        one of the official model names listed by `whisper.available_models()` or
        a local Pytorch checkpoint which is in the original OpenAI format
    download_root: str
        path to download the model files; by default, it uses "~/.cache/whisper"

    Returns
    -------
    model : Whisper
        The Whisper ASR model instance
    """

    if download_root is None:
        download_root = os.path.join(os.path.expanduser("~"), ".cache/whisper")

    weights, config, alignment_heads = load_torch_weights_and_config(
        name_or_path, download_root
    )
    dims = torch_whisper.ModelDimensions(**config)
    model = torch_whisper.Whisper(dims)
    model.load_state_dict(weights)

    if alignment_heads is not None:
        model.set_alignment_heads(alignment_heads)

    return model


def convert(name_or_path: str, dtype: mx.Dtype = mx.float16):
    def remap(key, value):
        key = key.replace("mlp.0", "mlp1")
        key = key.replace("mlp.2", "mlp2")
        if "conv" in key and value.ndim == 3:
            value = value.swapaxes(1, 2)
        if isinstance(value, torch.Tensor):
            value = mx.array(value.detach())
        return key, value.astype(dtype)

    weights, config, alignment_heads = load_torch_weights_and_config(name_or_path)
    weights.pop("encoder.positional_embedding", None)
    weights = dict(remap(k, v) for k, v in weights.items())

    model_dims = ModelDimensions(**config)
    model = Whisper(model_dims, dtype)
    model.load_weights(list(weights.items()), strict=False)

    if alignment_heads is not None:
        model.set_alignment_heads(alignment_heads)

    return model


def upload_to_hub(path: str, name: str, torch_name_or_path: str):
    import os

    from huggingface_hub import HfApi, ModelCard, logging

    repo_id = f"mlx-community/{name}"
    text = f"""
---
library_name: mlx
---

# {name}
This model was converted to MLX format from [`{torch_name_or_path}`]().

## Use with mlx
```bash
pip install mlx-whisper
```

```python
import mlx_whisper

result = mlx_whisper.transcribe(
    "FILE_NAME",
    path_or_hf_repo={repo_id},
)
```
"""
    card = ModelCard(text)
    card.save(os.path.join(path, "README.md"))

    logging.set_verbosity_info()

    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=path,
        repo_id=repo_id,
        repo_type="model",
    )


def quantize(weights, config, args):
    quantized_config = copy.deepcopy(config)

    # Load the model:
    model = Whisper(ModelDimensions(**config))
    weights = tree_map(mx.array, weights)
    model.update(tree_unflatten(list(weights.items())))

    # Quantize the model:
    nn.quantize(model, args.q_group_size, args.q_bits)

    # Update the config:
    quantized_config["quantization"] = {
        "group_size": args.q_group_size,
        "bits": args.q_bits,
    }
    quantized_weights = dict(tree_flatten(model.parameters()))

    return quantized_weights, quantized_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Whisper weights to MLX.")
    parser.add_argument(
        "--torch-name-or-path",
        type=str,
        default="tiny",
        help="The name or path to the PyTorch model.",
    )
    parser.add_argument(
        "--mlx-path",
        type=str,
        default="mlx_models",
        help="The path to save the MLX model.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="The dtype to save the MLX model.",
    )
    parser.add_argument(
        "-q",
        "--quantize",
        help="Generate a quantized model.",
        action="store_true",
    )
    parser.add_argument(
        "--q-group-size",
        help="Group size for quantization.",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--q-bits",
        help="Bits per weight for quantization.",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--upload-name",
        help="The name of model to upload to Hugging Face MLX Community",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    assert (
        args.dtype in _VALID_DTYPES
    ), f"dtype {args.dtype} not found in {_VALID_DTYPES}"
    dtype = getattr(mx, args.dtype)

    print("[INFO] Loading")
    model = convert(args.torch_name_or_path, dtype)
    config = asdict(model.dims)
    weights = dict(tree_flatten(model.parameters()))

    if args.quantize:
        print("[INFO] Quantizing")
        weights, config = quantize(weights, config, args)

    mlx_path = Path(args.mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    # Save weights
    print("[INFO] Saving")
    mx.save_safetensors(str(mlx_path / "weights.safetensors"), weights)

    # Save config.json with model_type
    with open(str(mlx_path / "config.json"), "w") as f:
        config["model_type"] = "whisper"
        json.dump(config, f, indent=4)

    if args.upload_name is not None:
        upload_to_hub(mlx_path, args.upload_name, args.torch_name_or_path)
