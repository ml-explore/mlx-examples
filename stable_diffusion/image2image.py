# Copyright © 2023 Apple Inc.

import argparse
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm

from stable_diffusion import StableDiffusion, StableDiffusionXL

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate images from an image and a textual prompt using stable diffusion"
    )
    parser.add_argument("image")
    parser.add_argument("prompt")
    parser.add_argument("--model", choices=["sd", "sdxl"], default="sdxl")
    parser.add_argument("--strength", type=float, default=0.9)
    parser.add_argument("--n_images", type=int, default=4)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--cfg", type=float)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--n_rows", type=int, default=1)
    parser.add_argument("--decoding_batch_size", type=int, default=1)
    parser.add_argument("--quantize", "-q", action="store_true")
    parser.add_argument("--no-float16", dest="float16", action="store_false")
    parser.add_argument("--preload-models", action="store_true")
    parser.add_argument("--output", default="out.png")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Load the models
    if args.model == "sdxl":
        sd = StableDiffusionXL("stabilityai/sdxl-turbo", float16=args.float16)

        if args.quantize:
            nn.quantize(
                sd.text_encoder_1, class_predicate=lambda _, m: isinstance(m, nn.Linear)
            )
            nn.quantize(
                sd.text_encoder_2, class_predicate=lambda _, m: isinstance(m, nn.Linear)
            )

            nn.quantize(sd.text_encoder_1)
            nn.quantize(sd.text_encoder_2)
            nn.quantize(sd.unet, group_size=32, bits=8)
        args.cfg = args.cfg or 0.0
        args.steps = args.steps or 2
    else:
        sd = StableDiffusion(
            "stabilityai/stable-diffusion-2-1-base", float16=args.float16
        )
        if args.quantize:
            nn.quantize(
                sd.text_encoder, class_predicate=lambda _, m: isinstance(m, nn.Linear)
            )
            nn.quantize(sd.unet, group_size=32, bits=8)
        args.cfg = args.cfg or 7.5
        args.steps = args.steps or 50

    # Fix the steps if they were set too low
    if int(args.steps * args.strength) < 1:
        args.steps = int(math.ceil(1 / args.strength))
        if args.verbose:
            print(
                f"Strength {args.strength} is too low so steps were set to {args.steps}"
            )

    # Ensure that models are read in memory if needed
    if args.preload_models:
        sd.ensure_models_are_loaded()

    # Read the image
    img = Image.open(args.image)

    # Make sure image shape is divisible by 64
    W, H = (dim - dim % 64 for dim in (img.width, img.height))
    if W != img.width or H != img.height:
        print(f"Warning: image shape is not divisible by 64, downsampling to {W}x{H}")
        img = img.resize((W, H), Image.NEAREST)  # use desired downsampling filter

    img = mx.array(np.array(img))
    img = (img[:, :, :3].astype(mx.float32) / 255) * 2 - 1

    # Noise and denoise the latents produced by encoding the img.
    latents = sd.generate_latents_from_image(
        img,
        args.prompt,
        strength=args.strength,
        n_images=args.n_images,
        cfg_weight=args.cfg,
        num_steps=args.steps,
        negative_text=args.negative_prompt,
    )
    for x_t in tqdm(latents, total=int(args.steps * args.strength)):
        mx.eval(x_t)

    # The following is not necessary but it may help in memory
    # constrained systems by reusing the memory kept by the unet and the text
    # encoders.
    if args.model == "sdxl":
        del sd.text_encoder_1
        del sd.text_encoder_2
    else:
        del sd.text_encoder
    del sd.unet
    del sd.sampler
    peak_mem_unet = mx.metal.get_peak_memory() / 1024**3

    # Decode them into images
    decoded = []
    for i in tqdm(range(0, args.n_images, args.decoding_batch_size)):
        decoded.append(sd.decode(x_t[i : i + args.decoding_batch_size]))
        mx.eval(decoded[-1])
    peak_mem_overall = mx.metal.get_peak_memory() / 1024**3

    # Arrange them on a grid
    x = mx.concatenate(decoded, axis=0)
    x = mx.pad(x, [(0, 0), (8, 8), (8, 8), (0, 0)])
    B, H, W, C = x.shape
    x = x.reshape(args.n_rows, B // args.n_rows, H, W, C).transpose(0, 2, 1, 3, 4)
    x = x.reshape(args.n_rows * H, B // args.n_rows * W, C)
    x = (x * 255).astype(mx.uint8)

    # Save them to disc
    im = Image.fromarray(np.array(x))
    im.save(args.output)

    # Report the peak memory used during generation
    if args.verbose:
        print(f"Peak memory used for the unet: {peak_mem_unet:.3f}GB")
        print(f"Peak memory used overall:      {peak_mem_overall:.3f}GB")
