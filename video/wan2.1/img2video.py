# Copyright © 2026 Apple Inc.

"""Generate videos from an image and text prompt using Wan2.1 I2V."""

import argparse
import logging

import mlx.core as mx
import mlx.nn as nn
from tqdm import tqdm
from wan import WanPipeline
from wan.utils import save_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate videos from an image and text prompt using Wan2.1 I2V"
    )
    parser.add_argument("prompt")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", choices=["i2v-14B"], default="i2v-14B")
    parser.add_argument(
        "--size",
        type=lambda x: tuple(map(int, x.split("x"))),
        default=(832, 480),
        help="Video size as WxH (default: 832x480)",
    )
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument(
        "--steps", type=int, default=40, help="Number of denoising steps"
    )
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--quantize",
        "-q",
        type=int,
        nargs="?",
        const=8,
        default=0,
        choices=[0, 4, 8],
        metavar="{4,8}",
        help="Quantize DiT weights (default: 8-bit when flag used without value)",
    )
    parser.add_argument(
        "--n-prompt",
        default="Text, watermarks, blurry image, JPEG artifacts",
    )
    parser.add_argument(
        "--teacache",
        type=float,
        default=0.0,
        help="TeaCache threshold for step skipping (0=off, 0.26=recommended for i2v)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to custom DiT weights (.safetensors), e.g. distilled models",
    )
    parser.add_argument(
        "--sampler",
        choices=["unipc", "euler"],
        default="unipc",
        help="Sampler: unipc (default) or euler (for step-distilled models)",
    )
    parser.add_argument("--output", default="out.mp4")
    parser.add_argument("--preload-models", action="store_true")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable Metal buffer cache (mx.set_cache_limit(0)) to reduce swap pressure",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.sampler == "euler":
        # Evenly spaced steps: e.g. 4 steps -> [1000, 750, 500, 250]
        n = args.steps
        denoising_step_list = [1000 * i // n for i in range(n, 0, -1)]
    else:
        denoising_step_list = None

    mx.set_default_device(mx.gpu)
    if args.no_cache:
        mx.set_cache_limit(0)

    if args.verbose:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger("wan").setLevel(logging.INFO)
        logging.getLogger("wan").addHandler(handler)

    # Load pipeline
    pipeline = WanPipeline(args.model, checkpoint=args.checkpoint)

    # Quantize DiT
    if args.quantize:
        nn.quantize(pipeline.flow, bits=args.quantize)
        print(f"Quantized DiT to {args.quantize}-bit")

    if args.preload_models:
        pipeline.ensure_models_are_loaded()

    # Generate latents (generator pattern)
    latents = pipeline.generate_latents(
        args.prompt,
        image_path=args.image,
        negative_prompt=args.n_prompt,
        size=args.size,
        frame_num=args.frames,
        num_steps=args.steps,
        guidance=args.guidance,
        shift=args.shift,
        seed=args.seed,
        teacache=args.teacache,
        verbose=args.verbose,
        denoising_step_list=denoising_step_list,
    )

    # 1. Conditioning
    conditioning = next(latents)
    mx.eval(conditioning)
    peak_mem_conditioning = mx.get_peak_memory() / 1024**3
    mx.reset_peak_memory()

    # Free T5 and CLIP memory
    del pipeline.t5
    if pipeline.clip is not None:
        del pipeline.clip
    mx.clear_cache()

    # 2. Denoising loop
    for x_t in tqdm(latents, total=args.steps):
        mx.eval(x_t)

    # Free DiT memory
    del pipeline.flow
    mx.clear_cache()
    peak_mem_generation = mx.get_peak_memory() / 1024**3
    mx.reset_peak_memory()

    # 3. VAE decode
    video = pipeline.decode(x_t)
    mx.eval(video)
    peak_mem_decoding = mx.get_peak_memory() / 1024**3

    # Save video
    save_video(video, args.output)

    if args.verbose:
        peak_mem_overall = max(
            peak_mem_conditioning, peak_mem_generation, peak_mem_decoding
        )
        print(f"Peak memory conditioning: {peak_mem_conditioning:.3f}GB")
        print(f"Peak memory generation:   {peak_mem_generation:.3f}GB")
        print(f"Peak memory decoding:     {peak_mem_decoding:.3f}GB")
        print(f"Peak memory overall:      {peak_mem_overall:.3f}GB")
