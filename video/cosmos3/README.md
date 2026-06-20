Cosmos 3 Nano
=============

[NVIDIA Cosmos 3](https://github.com/NVIDIA/cosmos) Nano (16B) text-to-video
and image-to-video generation on Apple Silicon via MLX. Cosmos 3 is a
world foundation model for **physical AI** — robotics, autonomous driving,
and industrial simulation. The model weights are downloaded from the
[Hugging Face Hub](https://huggingface.co/nvidia/Cosmos3-Nano).

| Config | RAM (bf16) | RAM (8-bit) |
|--------|-----------|------------|
| Cosmos3-Nano 16B | ~48 GB | ~24 GB |

> **Model scope:** Cosmos 3 was trained on robotics manipulation, autonomous
> driving, and industrial/factory environments. It produces strong physical
> motion for on-distribution inputs (dashcam driving, robot arms, factory
> floors) but does not generalize well to arbitrary creative prompts. This
> matches [NVIDIA's model card](https://huggingface.co/nvidia/Cosmos3-Nano).

Installation
------------

Install the dependencies:

```shell
pip install -r requirements.txt
```

Download the model weights (~32 GB at bf16):

```shell
hf download nvidia/Cosmos3-Nano --local-dir weights/Cosmos3-Nano
```

Saving videos as MP4 requires [ffmpeg](https://ffmpeg.org/) on your PATH.
If ffmpeg is not installed, output will be saved as GIF instead.

Usage
-----

> **Note:** The examples below use `--quantize` for 8-bit mode (~24 GB).
> Without `--quantize`, the model runs at bf16 and requires ~48 GB.

### Text-to-Video

Generate a video with an on-distribution prompt:

```shell
python txt2video.py 'A car driving through a suburban intersection on a sunny day' \
    --quantize --output out.mp4
```

Higher resolution:

```shell
python txt2video.py 'A delivery truck backing into a warehouse loading dock' \
    --size 832x480 --frames 16 --steps 30 --guidance 6.0 --seed 42 \
    --quantize --output out_480p.mp4
```

### Image-to-Video

Generate a video conditioned on an input image (provide your own JPEG/PNG):

```shell
python img2video.py 'A robot arm reaches toward a red block on a table' \
    --image your_image.jpg --quantize --output out_i2v.mp4
```

### Audio

Joint video+audio generation (stereo 48 kHz, muxed into MP4):

```shell
python txt2video.py 'A robot arm pushes a metal box across a table' \
    --enable-audio --quantize --output out_audio.mp4
```

### Quantization

Pass `--quantize` (or `-q`) to quantize the transformer weights to 8-bit,
reducing model weight memory from ~32 GB to ~16 GB (total runtime memory
is higher due to activations and VAE):

```shell
python txt2video.py 'A forklift moving pallets in a warehouse' \
    --quantize --output out_q8.mp4
```

### Disabling the cache

For additional memory savings at the expense of speed, use `--no-cache`:

```shell
python txt2video.py 'A robot arm sorting objects on a conveyor belt' \
    --quantize --no-cache --output out_nocache.mp4
```

### Options

- **Negative prompts:** `--n-prompt 'blurry, low quality'` (default: model's
  built-in negative prompt)
- **Guidance scale:** `--guidance 6.0` (default)
- **Denoising steps:** `--steps 30` (default)
- **Random seed:** `--seed 42`

For all options, use `python txt2video.py --help`.

Performance
-----------

Measured on M4 Max (128 GB), 8-bit quantized, 30 denoising steps:

| Resolution | Frames | Generation time | Peak memory |
|------------|--------|----------------|-------------|
| 256x256 | 16 | ~38s | ~17 GB |
| 480p (832x480) | 16 | ~252s | ~24 GB |
| 720p (1280x720) | 16 | ~591s | ~48 GB |

The pipeline caches text token KV pairs across denoising steps (text
embeddings are constant), which significantly reduces per-step compute
at lower resolutions.

Hardware requirements:

- **8-bit quantized, 256p:** ~17 GB peak (measured on M4 Max); 24 GB+ recommended
- **8-bit quantized, 480p:** ~24 GB peak; 48 GB+ recommended
- **bf16 full precision:** ~48 GB peak; 48 GB+ Mac (M4 Max or higher)

Architecture
------------

Cosmos 3 uses a Mixture-of-Transformers (MoT) design with two pathways:

- **Understanding (reasoner):** causal self-attention (Qwen3-VL text backbone)
- **Generation (diffuser):** full bidirectional attention for video/audio synthesis

Video VAE: Wan2.2 AutoencoderKL (16x spatial, 4x temporal downsampling).
Audio: Cosmos3 AVAEAudioTokenizer (Oobleck decoder, stereo 48 kHz).
Scheduler: UniPC multi-step predictor-corrector.

License
-------

Model weights are under [NVIDIA OpenMDW 1.1](https://openmdw.ai/license/1-1/)
(commercial and non-commercial use permitted).

References
----------

1. [NVIDIA Cosmos 3](https://github.com/NVIDIA/cosmos)
2. [Cosmos3-Nano model card](https://huggingface.co/nvidia/Cosmos3-Nano)
