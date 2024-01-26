# Copyright © 2023 Apple Inc.

import argparse

import mlx.core as mx
import numpy as np
from PIL import Image
from tqdm import tqdm

from stable_diffusion import StableDiffusion

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate images from a textual prompt using stable diffusion"
    )
    parser.add_argument("prompt")
    parser.add_argument("--n_images", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--n_rows", type=int, default=1)
    parser.add_argument("--decoding_batch_size", type=int, default=1)
    parser.add_argument("--output", default="out.png")
    args = parser.parse_args()

    sd = StableDiffusion()

    # Generate the latent vectors using diffusion
    latents = sd.generate_latents(
        args.prompt,
        n_images=args.n_images,
        cfg_weight=args.cfg,
        num_steps=args.steps,
        negative_text=args.negative_prompt,
    )
    for x_t in tqdm(latents, total=args.steps):
        mx.eval(x_t)

    # Decode them into images
    decoded = []
    for i in tqdm(range(0, args.n_images, args.decoding_batch_size)):
        decoded.append(sd.decode(x_t[i : i + args.decoding_batch_size]))
        mx.eval(decoded[-1])

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
