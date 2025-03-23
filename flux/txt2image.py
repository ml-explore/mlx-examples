# Copyright © 2024 Apple Inc.

import argparse

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm

from flux import FluxPipeline


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


def quantization_predicate(name, m):
    return hasattr(m, "to_quantized") and m.weight.shape[1] % 512 == 0


def load_adapter(flux, adapter_file, fuse=False):
    weights, lora_config = mx.load(adapter_file, return_metadata=True)
    rank = int(lora_config["lora_rank"])
    num_blocks = int(lora_config["lora_blocks"])
    flux.linear_to_lora_layers(rank, num_blocks)
    flux.flow.load_weights(list(weights.items()), strict=False)
    if fuse:
        flux.fuse_lora_layers()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate images from a textual prompt using FLUX"
    )
    parser.add_argument("prompt")
    parser.add_argument("--model", choices=["schnell", "dev"], default="schnell")
    parser.add_argument("--n-images", type=int, default=4)
    parser.add_argument(
        "--image-size", type=lambda x: tuple(map(int, x.split("x"))), default=(512, 512)
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--n-rows", type=int, default=1)
    parser.add_argument("--decoding-batch-size", type=int, default=1)
    parser.add_argument("--quantize", "-q", action="store_true")
    parser.add_argument("--preload-models", action="store_true")
    parser.add_argument("--output", default="out.png")
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--adapter")
    parser.add_argument("--fuse-adapter", action="store_true")
    parser.add_argument("--no-t5-padding", dest="t5_padding", action="store_false")
    parser.add_argument("--force-shard", action="store_true")
    args = parser.parse_args()

    # Load the models
    flux = FluxPipeline("flux-" + args.model, t5_padding=args.t5_padding)
    args.steps = args.steps or (50 if args.model == "dev" else 2)

    if args.adapter:
        load_adapter(flux, args.adapter, fuse=args.fuse_adapter)

    if args.quantize:
        nn.quantize(flux.flow, class_predicate=quantization_predicate)
        nn.quantize(flux.t5, class_predicate=quantization_predicate)
        nn.quantize(flux.clip, class_predicate=quantization_predicate)

    # Figure out what kind of distributed generation we should do
    group = mx.distributed.init()
    n_images = args.n_images
    should_gather = False
    if group.size() > 1:
        if args.force_shard or n_images < group.size() or n_images % group.size() != 0:
            flux.flow.shard(group)
        else:
            n_images //= group.size()
            should_gather = True

        # If we are sharding we should have the same seed and if we are doing
        # data parallel generation we should have different seeds
        if args.seed is None:
            args.seed = mx.distributed.all_sum(mx.random.randint(0, 2**20)).item()
        if should_gather:
            args.seed = args.seed + group.rank()

    if args.preload_models:
        flux.ensure_models_are_loaded()

    # Make the generator
    latent_size = to_latent_size(args.image_size)
    latents = flux.generate_latents(
        args.prompt,
        n_images=n_images,
        num_steps=args.steps,
        latent_size=latent_size,
        guidance=args.guidance,
        seed=args.seed,
    )

    # First we get and eval the conditioning
    conditioning = next(latents)
    mx.eval(conditioning)
    peak_mem_conditioning = mx.get_peak_memory() / 1024**3
    mx.reset_peak_memory()

    # The following is not necessary but it may help in memory constrained
    # systems by reusing the memory kept by the text encoders.
    del flux.t5
    del flux.clip

    # Actual denoising loop
    for x_t in tqdm(latents, total=args.steps, disable=group.rank() > 0):
        mx.eval(x_t)

    # The following is not necessary but it may help in memory constrained
    # systems by reusing the memory kept by the flow transformer.
    del flux.flow
    peak_mem_generation = mx.get_peak_memory() / 1024**3
    mx.reset_peak_memory()

    # Decode them into images
    decoded = []
    for i in tqdm(range(0, n_images, args.decoding_batch_size)):
        decoded.append(flux.decode(x_t[i : i + args.decoding_batch_size], latent_size))
        mx.eval(decoded[-1])
    peak_mem_decoding = mx.get_peak_memory() / 1024**3
    peak_mem_overall = max(
        peak_mem_conditioning, peak_mem_generation, peak_mem_decoding
    )

    # Gather them if each node has different images
    decoded = mx.concatenate(decoded, axis=0)
    if should_gather:
        decoded = mx.distributed.all_gather(decoded)
        mx.eval(decoded)

    if args.save_raw:
        *name, suffix = args.output.split(".")
        name = ".".join(name)
        x = decoded
        x = (x * 255).astype(mx.uint8)
        for i in range(len(x)):
            im = Image.fromarray(np.array(x[i]))
            im.save(".".join([name, str(i), suffix]))
    else:
        # Arrange them on a grid
        x = decoded
        x = mx.pad(x, [(0, 0), (4, 4), (4, 4), (0, 0)])
        B, H, W, C = x.shape
        x = x.reshape(args.n_rows, B // args.n_rows, H, W, C).transpose(0, 2, 1, 3, 4)
        x = x.reshape(args.n_rows * H, B // args.n_rows * W, C)
        x = (x * 255).astype(mx.uint8)

        # Save them to disc
        im = Image.fromarray(np.array(x))
        im.save(args.output)

    # Report the peak memory used during generation
    if args.verbose and group.rank() == 0:
        print(f"Peak memory used for the text:       {peak_mem_conditioning:.3f}GB")
        print(f"Peak memory used for the generation: {peak_mem_generation:.3f}GB")
        print(f"Peak memory used for the decoding:   {peak_mem_decoding:.3f}GB")
        print(f"Peak memory used overall:            {peak_mem_overall:.3f}GB")
