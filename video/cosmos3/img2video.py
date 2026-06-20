"""Generate videos from an input image using Cosmos 3 Nano on MLX."""

import argparse
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from cosmos3.load import load_transformer, load_tokenizer
from cosmos3.pipeline import Cosmos3GenerationPipeline, save_video
from cosmos3.decode_vae import decode_latents

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate videos from an input image using Cosmos 3 Nano"
    )
    parser.add_argument("prompt")
    parser.add_argument("--image", type=str, required=True, help="Conditioning image path")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="weights/Cosmos3-Nano",
        help="Path to Cosmos3-Nano weights directory",
    )
    parser.add_argument(
        "--size",
        type=lambda x: tuple(map(int, x.split("x"))),
        default=(256, 256),
        help="Video size as WxH (default: 256x256)",
    )
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument(
        "--steps", type=int, default=30, help="Number of denoising steps"
    )
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--quantize",
        "-q",
        type=int,
        nargs="?",
        const=8,
        default=0,
        choices=[0, 8],
        metavar="{8}",
        help="Quantize model weights to 8-bit (default when flag used without value)",
    )
    parser.add_argument(
        "--n-prompt",
        default=None,
        help="Negative prompt (default: model's built-in negative prompt)",
    )
    parser.add_argument("--output", default="out.mp4")
    parser.add_argument("--enable-audio", action="store_true", help="Generate audio")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable Metal buffer cache (mx.set_cache_limit(0)) to reduce swap pressure",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    width, height = args.size

    if args.frames < 8 or args.frames % 4 != 0:
        parser.error("--frames must be a multiple of 4 and >= 8 (e.g., 8, 16, 32)")
    if width % 16 != 0 or height % 16 != 0:
        parser.error(f"--size dimensions must be multiples of 16 (got {width}x{height})")

    mx.set_default_device(mx.gpu)
    if args.no_cache:
        mx.set_cache_limit(0)

    # Preflight: check required assets before expensive model load
    vae_dir = Path(args.model_dir) / "vae"
    if not (vae_dir / "config.json").exists():
        parser.error(f"VAE weights not found at {vae_dir}. Download the full model first.")
    if args.enable_audio and not (Path(args.model_dir) / "sound_tokenizer" / "config.json").exists():
        parser.error(f"Sound tokenizer not found. Download the full model or disable --enable-audio.")

    # Load conditioning image
    img = np.array(Image.open(args.image).convert("RGB"))
    print(f"Input image: {args.image} ({img.shape[1]}x{img.shape[0]})")

    # Load model
    print(f"Loading model from {args.model_dir}...")
    t0 = time.time()
    model = load_transformer(args.model_dir, reasoner_only=False)
    tokenizer = load_tokenizer(args.model_dir)

    if args.quantize:
        nn.quantize(model, bits=args.quantize)
        mx.eval(model.parameters())
        print(f"Quantized to {args.quantize}-bit")

    print(f"Model loaded in {time.time() - t0:.1f}s")

    pipeline = Cosmos3GenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        model_dir=args.model_dir,
    )

    # Generate latents
    print(f"\nGenerating {width}x{height} {args.frames}-frame i2v...")
    print(f"Prompt: {args.prompt}")
    t_start = time.time()

    result = pipeline.generate(
        prompt=args.prompt,
        num_frames=args.frames,
        height=height,
        width=width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        image=img,
        enable_audio=args.enable_audio,
        negative_prompt=args.n_prompt,
    )
    t_gen = time.time() - t_start

    # Free transformer memory before VAE decode
    del model
    del pipeline
    mx.clear_cache()

    if args.verbose and hasattr(mx, "get_peak_memory"):
        peak_mem_generation = mx.get_peak_memory() / 1024**3
        mx.reset_peak_memory()

    # Decode video
    vae_dir = str(Path(args.model_dir) / "vae")
    video = decode_latents(result["latents"], vae_dir)
    mx.eval(video)
    video_np = np.array(video[0].astype(mx.float32))
    video_np = (video_np * 255).clip(0, 255).astype(np.uint8)

    # Decode audio if generated
    audio_np = None
    if "audio_latents" in result:
        from cosmos3.decode_audio import decode_audio

        snd_dir = str(Path(args.model_dir) / "sound_tokenizer")
        audio_waveform = decode_audio(result["audio_latents"], snd_dir)
        mx.eval(audio_waveform)
        audio_np = np.array(audio_waveform[0].astype(mx.float32))

    # Save
    saved_path = save_video(video_np, args.output, fps=24, audio_waveform=audio_np)
    print(f"\nSaved to {saved_path} ({t_gen:.1f}s generation)")

    if args.verbose and hasattr(mx, "get_peak_memory"):
        peak_mem_decoding = mx.get_peak_memory() / 1024**3
        print(f"Peak memory generation: {peak_mem_generation:.3f}GB")
        print(f"Peak memory decoding:   {peak_mem_decoding:.3f}GB")
