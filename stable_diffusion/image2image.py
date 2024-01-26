# Copyright © 2023 Apple Inc.

import argparse

import mlx.core as mx
import numpy as np
from PIL import Image
from tqdm import tqdm

from stable_diffusion import StableDiffusion

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate images from an image and a textual prompt using stable diffusion"
    )
    parser.add_argument("image")
    parser.add_argument("prompt")
    parser.add_argument("--strength", type=float, default=0.9)
    parser.add_argument("--n_images", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--n_rows", type=int, default=1)
    parser.add_argument("--decoding_batch_size", type=int, default=1)
    parser.add_argument("--output", default="out.png")
    args = parser.parse_args()

    sd = StableDiffusion()

    # Read the image
    img = Image.open(args.image)

    # Make sure image shape is divisible by 64
    W, H = (dim - dim % 64 for dim in (img.width, img.height))
    if W != img.width or H != img.height:
        print(f"Warning: image shape is not divisible by 64, downsampling to {W}x{H}")
        img = img.resize((W, H), Image.NEAREST)  # use desired downsampling filter

    img = mx.array(np.array(img))
    img = (img[:, :, :3].astype(mx.float32) / 255) * 2 - 1

    # Noise and denoise the latents produced by encoding img.
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
