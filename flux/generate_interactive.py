import argparse

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm

from flux import FluxPipeline


def print_zero(group, *args, **kwargs):
    if group.rank() == 0:
        flush = kwargs.pop("flush", True)
        print(*args, **kwargs, flush=flush)


def quantization_predicate(name, m):
    return hasattr(m, "to_quantized") and m.weight.shape[1] % 512 == 0


def to_latent_size(image_size):
    h, w = image_size
    h = ((h + 15) // 16) * 16
    w = ((w + 15) // 16) * 16

    if (h, w) != image_size:
        print(
            "Warning: The image dimensions need to be divisible by 16px. "
            f"Changing size to {h}x{w}."
        )

    return (h // 8, w // 8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate images from a textual prompt using FLUX"
    )
    parser.add_argument("--quantize", "-q", action="store_true")
    parser.add_argument("--model", choices=["schnell", "dev"], default="schnell")
    parser.add_argument("--output", default="out.png")
    args = parser.parse_args()

    flux = FluxPipeline("flux-" + args.model, t5_padding=True)

    if args.quantize:
        nn.quantize(flux.flow, class_predicate=quantization_predicate)
        nn.quantize(flux.t5, class_predicate=quantization_predicate)
        nn.quantize(flux.clip, class_predicate=quantization_predicate)

    group = mx.distributed.init()
    if group.size() > 1:
        flux.flow.shard(group)

    print_zero(group, "Loading models")
    flux.ensure_models_are_loaded()

    def print_help():
        print_zero(group, "The command list:")
        print_zero(group, "- 'q' to exit")
        print_zero(group, "- 's HxW' to change the size of the image")
        print_zero(group, "- 'n S' to change the number of steps")
        print_zero(group, "- 'h' to print this help")

    print_zero(group, "FLUX interactive session")
    print_help()
    seed = 0
    size = (512, 512)
    latent_size = to_latent_size(size)
    steps = 50 if args.model == "dev" else 4
    while True:
        prompt = input(">> " if group.rank() == 0 else "")
        if prompt == "q":
            break
        if prompt == "h":
            print_help()
            continue
        if prompt.startswith("s "):
            size = tuple([int(xi) for xi in prompt[2:].split("x")])
            print_zero(group, "Setting the size to", size)
            latent_size = to_latent_size(size)
            continue
        if prompt.startswith("n "):
            steps = int(prompt[2:])
            print_zero(group, "Setting the steps to", steps)
            continue

        seed += 1
        latents = flux.generate_latents(
            prompt,
            n_images=1,
            num_steps=steps,
            latent_size=latent_size,
            guidance=4.0,
            seed=seed,
        )
        print_zero(group, "Processing prompt")
        mx.eval(next(latents))
        print_zero(group, "Generating latents")
        for xt in tqdm(latents, total=steps, disable=group.rank() > 0):
            mx.eval(xt)
        print_zero(group, "Generating image")
        xt = flux.decode(xt, latent_size)
        xt = (xt * 255).astype(mx.uint8)
        mx.eval(xt)
        im = Image.fromarray(np.array(xt[0]))
        im.save(args.output)
        print_zero(group, "Saved at", args.output, end="\n\n")
