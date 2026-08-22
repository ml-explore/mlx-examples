# Notice

**This folder (`stable_fast_3d/`) is licensed differently from the rest of this repository.** The rest of `mlx-examples` is MIT-licensed (see the repository root `LICENSE`); this folder is a Derivative Work of [Stability-AI/stable-fast-3d](https://github.com/Stability-AI/stable-fast-3d) and is licensed under the **Stability AI Community License** (see `stable_fast_3d/LICENSE`), which is not sublicensable to MIT terms.

This Stability AI Model is licensed under the Stability AI Community License, Copyright © Stability AI Ltd. All Rights Reserved.

Powered by Stability AI.

## Changes made in this Derivative Work

Reimplemented the full SF3D inference pipeline natively in [MLX](https://github.com/ml-explore/mlx) (Apple Silicon, unified memory):

- Neural forward pass (`camera_embedder.py`, `dinov2.py`, `backbone.py`, `post_processor.py`, `decoder.py`): camera embedder, DINOv2 image tokenizer with per-layer AdaLN modulation, two-stream interleaved transformer backbone, pixel-shuffle post-processor, triplane query sampler, material decoder. Validated module-by-module against the original PyTorch model to float32 tolerance.
- Mesh finishing (`isosurface.py`, `uv_unwrap.py`, `texture_baker.py`): marching-tetrahedra isosurface extraction, UV unwrapping, texture baking, reimplemented in MLX/numpy (previously PyTorch/CPU-only).

No model weights are redistributed here; `weights.py` downloads and converts the checkpoint from the gated [stabilityai/stable-fast-3d](https://huggingface.co/stabilityai/stable-fast-3d) Hugging Face repo at runtime, and downloads the fixed tetrahedra grid (`load/tets/160_tets.npz`, a Stability AI-authored geometric asset, not a trained weight) from the upstream GitHub repo. Both remain under the original license terms.

Standalone reference implementation (development history, module-by-module validation scripts, more detail): https://github.com/bahaehmimdi/stable-fast-3d-mlx
