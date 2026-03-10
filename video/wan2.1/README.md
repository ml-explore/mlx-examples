Wan2.1
======

Wan2.1 text-to-video and image-to-video implementation in MLX. The model
weights are downloaded directly from the [Hugging Face
Hub](https://huggingface.co/Wan-AI).

| Model | Task | HF Repo | RAM (quantized) |
|-------|------|---------|-----------------|
| 1.3B | T2V | [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | ~16GB |
| 14B | T2V | [Wan-AI/Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) | ~48GB |
| 14B | I2V | [Wan-AI/Wan2.1-I2V-14B-480P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) | ~48GB |

![WAN 1.3B](static/out.mp4)

Installation
------------

Install the dependencies:

    pip install -r requirements.txt

> [!Note]
> Saving videos requires [ffmpeg](https://ffmpeg.org/) on your PATH.

Usage
-----

### Text-to-Video

Generate a video with the default 1.3B model:

```shell
python txt2video.py 'A cat playing piano' --output out.mp4 --verbose
```

Use the 14B model with quantization:

```shell
python txt2video.py 'A cat playing piano' \
    --model t2v-14B --quantize --output out_14B.mp4
```

Adjust resolution, frame count, and sampling parameters:

```shell
python txt2video.py 'Ocean waves crashing on a rocky shore at sunset' \
    --size 832x480 --frames 81 --steps 50 --guidance 5.0 --seed 42 \
    --output waves.mp4
```

For more parameters, use `python txt2video.py --help`.

### Image-to-Video

Generate a video from an input image:

```shell
python img2video.py 'A cat playing piano' \
    --image cat.jpg --quantize --output out_i2v.mp4
```

Adjust resolution and sampling parameters:

```shell
python img2video.py 'Ocean waves crashing on a rocky shore at sunset' \
    --image shore.jpg --size 832x480 --frames 81 --steps 40 \
    --guidance 5.0 --shift 3.0 --seed 42 --output waves_i2v.mp4
```

For more parameters, use `python img2video.py --help`.

### Quantization

Pass `--quantize` (or `-q`) to the CLI, or apply it directly:

```python
import mlx.nn as nn

pipeline = WanT2VPipeline("t2v-1.3B")
nn.quantize(pipeline.flow, class_predicate=lambda n, m: (
    hasattr(m, "to_quantized") and m.weight.shape[1] % 512 == 0
))
```

### Custom DiT Weights

Use `--checkpoint` to load custom DiT weights (e.g. [step-distilled models](https://huggingface.co/lightx2v/Wan2.1-Distill-Models)).
Pass `--sampler euler` to use Euler sampling for step-distilled models:

For text to video pipeline you can try [this 4 steps distilled model](https://huggingface.co/lightx2v/Wan2.1-Distill-Models/blob/main/wan2.1_t2v_14b_lightx2v_4step.safetensors)
```shell
python txt2video.py 'A cat playing piano' \
    --model t2v-14B --checkpoint /path/to/distilled_dit.safetensors \
    --sampler euler --steps 4 --guidance 1.0 \
    --quantize --output out_distilled.mp4
```

For image to video pipeline we use [4 steps distilled i2v model](https://huggingface.co/lightx2v/Wan2.1-Distill-Models/blob/main/wan2.1_i2v_480p_scaled_fp8_e4m3_lightx2v_4step.safetensors)
```shell
python img2video.py 'A cat playing piano' \
    --image cat.jpg --checkpoint /path/to/distilled_i2v.safetensors \
    --sampler euler --steps 4 --guidance 1.0 --shift 5.0 \
    --quantize --output out_i2v_distilled.mp4
```

### Options

- **Negative prompts**: `--n-prompt 'blurry, low quality, distorted'`
- **Disable CFG**: `--guidance 1.0` skips the unconditional pass, roughly
  halving compute per step.

### TeaCache

TeaCache skips redundant transformer computations when consecutive steps
produce similar embeddings, eliminating 20-60% of forward passes.

```shell
python txt2video.py 'A cat playing piano' --teacache 0.05 --output out.mp4
```

Pass `--no-ret-steps` to use the raw time embedding instead of the default
projected embedding.

Recommended thresholds (1.3B):

| Threshold | Skip Rate | Quality |
|-----------|-----------|---------|
| `0.05` | ~34% | Almost lossless |
| `0.1` | ~50% | Slightly corrupted |
| `0.26` | ~75% | Visible quality loss |
