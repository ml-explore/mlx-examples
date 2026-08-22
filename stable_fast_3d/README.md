Stable Fast 3D
==============

[Stable Fast 3D (SF3D)](https://github.com/Stability-AI/stable-fast-3d) in
MLX. SF3D is Stability AI's feedforward single-image -> textured 3D mesh
model. This is a native MLX reimplementation of the full inference pipeline
- no PyTorch, no MPS fallback.

> **License note:** unlike the rest of `mlx-examples` (MIT), this folder is
> licensed under the **Stability AI Community License** - see `LICENSE` and
> `NOTICE.md`. The underlying model is also gated on Hugging Face; you must
> request access before you can download the checkpoint.

Why a separate port: the official PyTorch implementation's MPS backend on
Apple Silicon is slow (full model reload + inference on every generation).
On an M1 Pro, this MLX port brings a single generation from **7-9 minutes**
down to **~12.4 seconds** end-to-end (~34-44x), with the neural forward
pass and mesh finishing (UV unwrap, texture baking) both running in
100% MLX/numpy.

Installation
------------

```
pip install -r requirements.txt
```

The model is gated at [Hugging Face](https://huggingface.co/stabilityai/stable-fast-3d):

1. Request access [here](https://huggingface.co/stabilityai/stable-fast-3d).
2. Create a read-access token [here](https://huggingface.co/settings/tokens).
3. `huggingface-cli login` and enter the token.

Usage
-----

```
python image_to_3d.py chair.png output.glb
```

The checkpoint (~4GB) is downloaded and converted to MLX format on first
run, then cached at `~/.cache/mlx-stable-fast-3d/`.

Or from Python:

```python
from PIL import Image
from stable_fast_3d import StableFast3D

model = StableFast3D()  # loads weights once
model.run(Image.open("chair.png"), "output.glb")
```

`StableFast3D.run` accepts `foreground_ratio`, `remove_bg` (background
removal via `rembg`; set `False` if your image already has a clean alpha
channel), and `bake_resolution` (texture resolution, default 512).

What's not ported
------------------

- **CLIP-based material estimation** (`image_estimator` in the original) -
  roughness/metallic use fixed constants instead of a real per-image
  estimate. Not needed for a raw textured mesh.
- **Illumination estimation** (`global_estimator`) - opt-in in the
  original, not requested by default, not ported.
- **Remeshing** (quad/triangle) - the original repo's optional
  post-processing step; out of scope here, use the original repo's
  `run.py --remesh_option` if you need it.

Speed comparison
-----------------

Measured on an M1 Pro, single generation, model already loaded:

| Version | Backend | Time | Detail |
|---|---|---|---|
| Original SF3D | PyTorch/CUDA (reference hardware) | 7-9 min | Full model reload every generation, no Mac-specific optimization |
| This port, neural pass only | MLX (network) + PyTorch/MPS (finishing) | ~48s | 0.5s weight load + 5.3s MLX inference + 40.8s torch/MPS subprocess (isosurface, UV, texture) |
| This port, full pipeline | 100% MLX + numpy | ~12.4s | 0.4s weight load + 5.4s neural pass + 7.0s MLX/numpy finishing |

Validation
----------

Each module was validated against the original PyTorch model's intermediate
tensors to float32 tolerance (max-abs-diff < 1e-4 throughout the neural
pass), then checked end-to-end: feeding this port's `scene_codes` into the
original (unmodified) PyTorch mesh-extraction code produces an identical
mesh (same vertex/face counts, vertex positions matching to 2.2e-5 max /
1.8e-7 mean) as a mesh produced by a fresh, real PyTorch forward pass on
the same image. Full module-by-module validation scripts and development
history: https://github.com/bahaehmimdi/stable-fast-3d-mlx.

Citation
--------

```BibTeX
@article{sf3d2024,
  title={SF3D: Stable Fast 3D Mesh Reconstruction with UV-unwrapping and Illumination Disentanglement},
  author={Boss, Mark and Huang, Zixuan and Vasishta, Aaryaman and Jampani, Varun},
  journal={arXiv preprint},
  year={2024}
}
```
