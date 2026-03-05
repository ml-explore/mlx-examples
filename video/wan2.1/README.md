Wan2.1
======

Wan2.1 text-to-video implementation in MLX. The model weights are downloaded
directly from the [Hugging Face Hub](https://huggingface.co/Wan-AI).

Two model sizes are supported:

| Model | Parameters | HF Repo | RAM (quantized) |
|-------|-----------|---------|-----------------|
| 1.3B | 1.3B | [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | ~16GB |
| 14B | 14B | [Wan-AI/Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) | ~48GB |

![WAN 1.3B](static/out.mp4)

Installation
------------

The dependencies are minimal, namely:

- `huggingface-hub` to download the checkpoints.
- `tokenizers` for the T5 tokenizer
- `einops` for tensor reshaping
- `tqdm` and `numpy` for the scripts
- `mlx >= 0.22.0`

You can install all of the above with the `requirements.txt` as follows:

    pip install -r requirements.txt

> [!Note]
> Saving videos requires [ffmpeg](https://ffmpeg.org/) to be installed and
> available on your PATH.

Usage
-----

Generate a video with the default 1.3B model:

```shell
python txt2video.py 'A cat playing piano' \
    --output out.mp4 \
    --verbose
```

Use the 14B model with quantization:

```shell
python txt2video.py 'A cat playing piano' \
    --model t2v-14B \
    --quantize \
    --output out_14B.mp4 \
    --verbose
```

Adjust resolution, frame count, and sampling parameters:

```shell
python txt2video.py 'Ocean waves crashing on a rocky shore at sunset' \
    --size 832x480 \
    --frames 81 \
    --steps 50 \
    --guidance 5.0 \
    --seed 42 \
    --output waves.mp4
```

For more parameters, use the `--help` command:

```shell
python txt2video.py --help
```

Inference
---------

The `WanT2VPipeline` class follows the same generator pattern as
`FluxPipeline`. This allows fine-grained control over memory by unloading
models between stages.

```python
import mlx.core as mx
from wan import WanT2VPipeline

# This will download all the weights from HF Hub
pipeline = WanT2VPipeline("t2v-1.3B")

# Optionally specify dtype (default: mx.bfloat16)
# pipeline = WanT2VPipeline("t2v-1.3B", dtype=mx.float16)

# Make a generator that returns the latent variables from the reverse
# diffusion process
latents = pipeline.generate_latents(
    "A cat playing piano",
    num_steps=50,
    size=(832, 480),
    frame_num=81,
)

# The first yield contains the conditioning (noise + text embeddings).
# Evaluating it here allows us to unload T5 before running the DiT.
conditioning = next(latents)
mx.eval(conditioning)

# Free T5 memory (~4GB)
del pipeline.t5

# Evaluate each denoising step
for x_t in latents:
    mx.eval(x_t)

# Free DiT memory
del pipeline.flow

# Decode latents to video frames
video = pipeline.decode(x_t)
mx.eval(video)

# Save to file
from wan.utils import save_video
save_video(video, "out.mp4")
```

### Quantization

Quantization reduces memory usage significantly. Pass `--quantize` (or `-q`)
to the CLI, or apply it directly:

```python
import mlx.nn as nn

pipeline = WanT2VPipeline("t2v-1.3B")
nn.quantize(pipeline.flow, class_predicate=lambda n, m: (
    hasattr(m, "to_quantized") and m.weight.shape[1] % 512 == 0
))
```

### Negative Prompts

Use `--n-prompt` to guide the model away from unwanted content:

```shell
python txt2video.py 'A serene mountain landscape' \
    --n-prompt 'blurry, low quality, distorted' \
    --output landscape.mp4
```

### Disabling Classifier-Free Guidance

Set `--guidance 1.0` to skip the unconditional forward pass, roughly halving
the compute per denoising step:

```shell
python txt2video.py 'A cat playing piano' \
    --guidance 1.0 \
    --output out_no_cfg.mp4
```
