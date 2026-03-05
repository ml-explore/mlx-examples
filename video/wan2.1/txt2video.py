# Copyright © 2025 Apple Inc.

"""Generate videos from text using Wan2.1."""

import argparse

import mlx.core as mx
import mlx.nn as nn
from tqdm import tqdm
from wan import WanT2VPipeline
from wan.utils import save_video


def quantization_predicate(name, m):
    return hasattr(m, "to_quantized") and m.weight.shape[1] % 512 == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate videos from text using Wan2.1"
    )
    parser.add_argument("prompt")
    parser.add_argument("--model", choices=["t2v-1.3B", "t2v-14B"], default="t2v-1.3B")
    parser.add_argument(
        "--size",
        type=lambda x: tuple(map(int, x.split("x"))),
        default=(832, 480),
        help="Video size as WxH (default: 832x480)",
    )
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=5.0)
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
    parser.add_argument("--n-prompt", default="")
    parser.add_argument("--output", default="out.mp4")
    parser.add_argument("--preload-models", action="store_true")
    parser.add_argument(
        "--compile-vae", action="store_true", help="Compile VAE decoder"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)

    # Load pipeline
    pipeline = WanT2VPipeline(args.model)

    # Quantize DiT
    if args.quantize:
        nn.quantize(
            pipeline.flow, bits=args.quantize, class_predicate=quantization_predicate
        )
        print(f"Quantized DiT to {args.quantize}-bit")

    if args.preload_models:
        pipeline.ensure_models_are_loaded()

    # Generate latents (generator pattern matching flux)
    latents = pipeline.generate_latents(
        args.prompt,
        negative_prompt=args.n_prompt,
        size=args.size,
        frame_num=args.frames,
        num_steps=args.steps,
        guidance=args.guidance,
        shift=args.shift,
        seed=args.seed,
    )

    # 1. Conditioning
    conditioning = next(latents)
    mx.eval(conditioning)
    peak_mem_conditioning = mx.get_peak_memory() / 1024**3
    mx.reset_peak_memory()

    # Free T5 memory
    del pipeline.t5
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
    video = pipeline.decode(x_t, compile_vae=args.compile_vae)
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
