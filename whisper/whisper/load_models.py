# Copyright © 2023 Apple Inc.

import hashlib
import os
import urllib
import warnings
from typing import List

import mlx.core as mx
import torch
from mlx.utils import tree_map
from tqdm import tqdm

from . import torch_whisper, whisper

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

    model_bytes = open(download_target, "rb").read()
    if hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model."
        )

    return download_target


def available_models() -> List[str]:
    """Returns the names of available models"""
    return list(_MODELS.keys())


def load_torch_model(
    name: str,
    download_root: str = None,
) -> torch_whisper.Whisper:
    """
    Load a Whisper ASR model

    Parameters
    ----------
    name : str
        one of the official model names listed by `whisper.available_models()`
    download_root: str
        path to download the model files; by default, it uses "~/.cache/whisper"

    Returns
    -------
    model : Whisper
        The Whisper ASR model instance
    """

    if download_root is None:
        download_root = os.path.join(os.path.expanduser("~"), ".cache/whisper")

    if name in _MODELS:
        checkpoint_file = _download(_MODELS[name], download_root)
        alignment_heads = _ALIGNMENT_HEADS[name]
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    with open(checkpoint_file, "rb") as fp:
        checkpoint = torch.load(fp)

    dims = torch_whisper.ModelDimensions(**checkpoint["dims"])
    model = torch_whisper.Whisper(dims)
    model.load_state_dict(checkpoint["model_state_dict"])

    if alignment_heads is not None:
        model.set_alignment_heads(alignment_heads)

    return model


def convert(model, rules=None):
    params = {}
    if rules is not None and type(model) in rules:
        out = rules[type(model)](model, rules)
        return out
    if isinstance(model, torch.Tensor):
        return mx.array(model.detach().numpy())
    if isinstance(model, torch.nn.ModuleList):
        return [convert(n, rules) for n in model.children()]
    if isinstance(model, torch.nn.Conv1d):
        return {
            "weight": convert(model.weight).transpose(0, 2, 1),
            "bias": convert(model.bias),
        }
    for k, n in model.named_children():
        if k in rules:
            params.update(rules[k](n, rules))
        else:
            params[k] = convert(n, rules)
    for k, p in model.named_parameters(recurse=False):
        params[k] = convert(p)
    return params


def torch_to_mlx(
    torch_model: torch_whisper.Whisper,
    dtype: mx.Dtype = mx.float16,
) -> whisper.Whisper:
    def convert_rblock(model, rules):
        children = dict(model.named_children())
        mlp = list(children.pop("mlp").children())
        params = {
            "mlp1": convert(mlp[0], rules),
            "mlp2": convert(mlp[-1], rules),
        }
        for k, n in children.items():
            params[k] = convert(n, rules)
        return params

    rules = {
        torch_whisper.ResidualAttentionBlock: convert_rblock,
    }

    params = convert(torch_model, rules)

    mlx_model = whisper.Whisper(torch_model.dims, dtype)
    params = tree_map(lambda p: p.astype(dtype), params)
    mlx_model.update(params)
    return mlx_model


def load_model(
    name: str,
    download_root: str = None,
    dtype: mx.Dtype = mx.float32,
) -> whisper.Whisper:
    return torch_to_mlx(load_torch_model(name, download_root), dtype)
