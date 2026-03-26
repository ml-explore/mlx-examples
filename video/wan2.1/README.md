Wan2.1
======

Wan2.1 text-to-video and image-to-video implementation in MLX. The model
weights are downloaded directly from the [Hugging Face
Hub](https://huggingface.co/Wan-AI).

| Model | Task | HF Repo | RAM (unquantized), 81 frames | Single DiT step on M4 Max chip, 81 frames |
|-------|------|---------|-----------------|---|
| 1.3B | T2V | [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | ~10GB | ~90 s/it |
| 14B | T2V | [Wan-AI/Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) | ~36GB | ~230 s/it |
| 14B | I2V | [Wan-AI/Wan2.1-I2V-14B-480P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) | ~39GB | ~250 s/it |

| T2V 1.3B | T2V 14B | I2V 14B |
|---|---|---|
| ![WAN t2v 1.3B](static/out_t2v_1_3b.gif) |![WAN t2v 14B distilled](static/out_t2v_cats.gif) | ![WAN t2v 14B distilled](static/out_i2v_astronaut.gif) |
| Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage. | Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage. | An astronaut riding a horse |

Installation
------------

Install the dependencies:
```shell
pip install -r requirements.txt
```

Saving videos requires [ffmpeg](https://ffmpeg.org/) on your PATH.

Usage
-----

### Text-to-Video

Generate a video with the default 1.3B model:

```shell
python txt2video.py 'A cat playing piano' --output out.mp4
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
python img2video.py 'Astronaut riding a horse' \
    --image ./inputs/astronaut-on-a-horse.png --quantize --output out_i2v.mp4
```

Adjust resolution and sampling parameters:

```shell
python img2video.py 'Astronaut riding a horse' \
    --image ./inputs/astronaut-on-a-horse.png --size 832x480 --frames 81 --steps 40 \
    --guidance 5.0 --shift 3.0 --seed 42 --output out_i2v.mp4
```

For more parameters, use `python img2video.py --help`.

### Quantization

Pass `--quantize` (or `-q`) to the CLI

```shell
python txt2video.py 'A cat playing piano' --quantize --output out_quantized.mp4
```

### Disabling the cache
To get additional memory savings at the expense of a bit of speed use `--no-cache` argument. It will prevent MLX from utilizing the cache (sets `mx.set_cache_limit(0)` under the hood). See [documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_cache_limit.html) for more info
```shell
python txt2video.py 'A cat playing piano' --output out.mp4 --no-cache
```

For 1.3B model 480p 81 frames `--no-cache` run utilizes ~10GB of RAM and ~14GB of RAM otherwise 

### Custom DiT Weights

Use `--checkpoint` to load custom DiT weights (e.g. [step-distilled models](https://huggingface.co/lightx2v/Wan2.1-Distill-Models)).
Pass `--sampler euler` to use Euler sampling for step-distilled models:

For text to video pipeline you can try [this 4 steps distilled model](https://huggingface.co/lightx2v/Wan2.1-Distill-Models/blob/main/wan2.1_t2v_14b_lightx2v_4step.safetensors)

```shell
wget https://huggingface.co/lightx2v/Wan2.1-Distill-Models/resolve/main/wan2.1_t2v_14b_lightx2v_4step.safetensors
```

```shell
python txt2video.py 'A cat playing piano' \
    --model t2v-14B --checkpoint ./wan2.1_t2v_14b_lightx2v_4step.safetensors \
    --sampler euler --steps 4 --guidance 1.0 \
    --quantize --output out_t2v_distilled.mp4
```

For image to video pipeline we use [4 steps distilled i2v model](https://huggingface.co/lightx2v/Wan2.1-Distill-Models/resolve/main/wan2.1_i2v_480p_lightx2v_4step.safetensors)

```shell
wget https://huggingface.co/lightx2v/Wan2.1-Distill-Models/resolve/main/wan2.1_i2v_480p_lightx2v_4step.safetensors
```

```shell
python img2video.py 'Astronaut riding a horse' \
    --image ./inputs/astronaut-on-a-horse.png --checkpoint ./wan2.1_i2v_480p_lightx2v_4step.safetensors \
    --sampler euler --steps 4 --guidance 1.0 --shift 5.0 \
    --quantize --output out_i2v_distilled.mp4
```

### Options

- **Negative prompts**: `--n-prompt 'blurry, low quality, distorted'`
- **Disable CFG**: `--guidance 1.0` skips the unconditional pass, roughly
  halving compute per step.

### TeaCache

[TeaCache](https://arxiv.org/abs/2411.19108) skips redundant transformer computations when consecutive steps
produce similar embeddings, eliminating 20-60% of forward passes. Note that the TeaCache parameters are calibrated for each resolution, consult with [LightX2V](https://github.com/ModelTC/LightX2V/tree/main/configs/caching) configs for advanced tweaking. Our defaults are located at [pipeline.py](./wan/pipeline.py#20)

```shell
python txt2video.py 'A cat playing piano' --teacache 0.05 --output out.mp4 --verbose
```

Recommended thresholds (1.3B):

| Threshold | Skip Rate | Quality |
|-----------|-----------|---------|
| `0.05` | ~34% | Almost lossless |
| `0.1` | ~58% | Slightly corrupted |
| `0.25` | ~76% | Visible quality loss |

#### Result with --teacache for 1.3B model
`Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.`
|`--teacache 0.05`, 34% steps skipped (17/50) |`--teacache 0.1`, 58% steps skipped (29/50) |`--teacache 0.25`, 76% steps skipped (38/50) |
|---|---|---|
|![WAN t2v 1.3B teacache=0.05](static/out_t2v_1_3b_teacache_005.gif)|![WAN t2v 1.3B teacache=0.05](static/out_t2v_1_3b_teacache_01.gif)|![WAN t2v 1.3B teacache=0.05](static/out_t2v_1_3b_teacache_025.gif)|

# References
1. [Original WAN 2.1 implementation](https://github.com/Wan-Video/Wan2.1)
2. [LightX2V](https://github.com/ModelTC/LightX2V)
