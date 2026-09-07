# DINOv2

[dinov2-mlx](https://github.com/bahaehmimdi/dinov2-mlx) runs
[DINOv2](https://github.com/facebookresearch/dinov2) on Apple Silicon,
verified against real, unmodified PyTorch at every stage (preprocessing,
model forward, both fp32 and fp16).

**Note on approach**: unlike the other examples in this repo, this does
not reimplement DINOv2 natively in `mlx.nn`. It runs the real,
unmodified `transformers.AutoModel` (`Dinov2Model`) through
[torch-mlx](https://github.com/bahaehmimdi/torch-mlx) — a from-scratch
`torch`-API-compatible layer backed by `mlx.core` — so the model
definition itself is real `transformers` code, not a hand-written port.
Given that's a different shape of contribution than this repo's usual
convention (native model + `torch` used only for offline weight
conversion), this entry is a link to the full implementation and
results rather than inline code — happy to restructure this however
maintainers think fits best, or to close this if it's not a good fit
for the example gallery.

## Results

Verified against real PyTorch: `pooler_output` matches to ~0.32%
relative error. Beats real PyTorch's own MPS backend by ~1.12-1.15x
across tested image sizes (both backends timed on the identical full
pipeline: image → resize → crop → forward). Full write-up, including
what didn't work, in the linked repo's `BENCHMARK_RESULTS.md`.

```python
from dinov2_mlx import DINOv2MLX
from PIL import Image

model = DINOv2MLX()
out = model.extract_features(Image.open("photo.png"))
out["pooler_output"]        # (1, 1024) numpy array
out["last_hidden_state"]    # (1, 257, 1024) numpy array
```

See [the repo](https://github.com/bahaehmimdi/dinov2-mlx) for install
instructions, the fp16/`mx.compile` options, and full benchmark
methodology.
