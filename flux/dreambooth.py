import argparse
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.nn.utils import average_gradients
from mlx.utils import tree_map, tree_reduce, tree_unflatten
from PIL import Image
from tqdm import tqdm

from flux import FluxPipeline
from flux.lora import LoRALinear


def linear_to_lora_layers(flux, args):
    lora_layers = []
    rank = args.lora_rank
    for name, mod in flux.flow.named_modules():
        if ".img_attn" not in name and ".txt_attn" not in name:
            continue
        if ".qkv" in name or ".proj" in name:
            lora_layers.append((name, LoRALinear.from_base(mod, r=rank)))
    flux.flow.update_modules(tree_unflatten(lora_layers))


def extract_latent_vectors(flux, image_folder):
    flux.ae.eval()
    latents = []
    for image in tqdm(Path(image_folder).iterdir()):
        img = Image.open(image)
        img = mx.array(np.array(img))
        img = (img[:, :, :3].astype(flux.dtype) / 255) * 2 - 1
        x_0 = flux.ae.encode(img[None])
        x_0 = x_0.astype(flux.dtype)
        mx.eval(x_0)
        latents.append(x_0)
    return mx.concatenate(latents)


def decode_latents(flux, x):
    decoded = []
    for i in tqdm(range(len(x))):
        decoded.append(flux.decode(x[i : i + 1]))
        mx.eval(decoded[-1])
    return mx.concatenate(decoded, axis=0)


def generate_latents(flux, n_images, prompt, steps, seed=None, leave=True):
    latents = flux.generate_latents(
        prompt,
        n_images=n_images,
        num_steps=steps,
        seed=seed,
    )
    for x_t in tqdm(latents, total=args.progress_steps, leave=leave):
        mx.eval(x_t)

    return x_t


def iterate_batches(t5_tokens, clip_tokens, x, batch_size):
    while True:
        indices = mx.random.randint(0, len(x), (batch_size,))
        t5_i = t5_tokens[indices]
        clip_i = clip_tokens[indices]
        x_i = x[indices]
        yield t5_i, clip_i, x_i


def generate_progress_images(iteration, flux, args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"out_{iteration:03d}.png"
    print(f"Generating {str(out_file)}")
    # Generate the latent vectors using diffusion
    n_images = 4
    latents = generate_latents(
        flux,
        n_images,
        args.progress_prompt,
        args.progress_steps,
        seed=42 + mx.distributed.init().rank(),
    )

    # Arrange them on a grid
    n_rows = 2
    x = decode_latents(flux, latents)
    x = mx.pad(x, [(0, 0), (4, 4), (4, 4), (0, 0)])
    B, H, W, C = x.shape
    x = x.reshape(n_rows, B // n_rows, H, W, C).transpose(0, 2, 1, 3, 4)
    x = x.reshape(n_rows * H, B // n_rows * W, C)
    x = mx.pad(x, [(4, 4), (4, 4), (0, 0)])
    x = (x * 255).astype(mx.uint8)

    # Save them to disc
    im = Image.fromarray(np.array(x))
    im.save(out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finetune Flux to generate images with a specific subject"
    )

    parser.add_argument(
        "--model",
        default="dev",
        choices=[
            "dev",
            "schnell",
        ],
        help="Which flux model to train",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=400,
        help="How many iterations to train for",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size to use when training the stable diffusion model",
    )
    parser.add_argument(
        "--progress-prompt",
        help="Use this prompt when generating images for evaluation",
    )
    parser.add_argument(
        "--progress-steps",
        type=int,
        default=50,
        help="Use this many steps when generating images for evaluation",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Generate images every PROGRESS_EVERY steps",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Save the model every CHECKPOINT_EVERY steps",
    )
    parser.add_argument(
        "--lora-rank", type=int, default=32, help="LoRA rank for finetuning"
    )
    parser.add_argument(
        "--learning-rate", type=float, default="1e-6", help="Learning rate for training"
    )
    parser.add_argument(
        "--grad-accumulate",
        type=int,
        default=1,
        help="Accumulate gradients for that many iterations before applying them",
    )
    parser.add_argument(
        "--output-dir", default="mlx_output", help="Folder to save the checkpoints in"
    )

    parser.add_argument("prompt")
    parser.add_argument("image_folder")

    args = parser.parse_args()

    args.progress_prompt = args.progress_prompt or args.prompt

    flux = FluxPipeline("flux-" + args.model)
    flux.ensure_models_are_loaded()
    flux.flow.freeze()
    linear_to_lora_layers(flux, args)

    trainable_params = tree_reduce(
        lambda acc, x: acc + x.size, flux.flow.trainable_parameters(), 0
    )
    print(f"Training {trainable_params / 1024**2:.3f}M parameters")

    optimizer = optim.Adam(learning_rate=args.learning_rate)
    state = [flux.flow.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def single_step(t5_tokens, clip_tokens, x, guidance):
        loss, grads = nn.value_and_grad(flux.flow, flux.training_loss)(
            t5_tokens, clip_tokens, x, guidance
        )
        grads = average_gradients(grads)
        optimizer.update(flux.flow, grads)

        return loss

    @partial(mx.compile, inputs=state, outputs=state)
    def compute_loss_and_grads(t5_tokens, clip_tokens, x, guidance):
        return nn.value_and_grad(flux.flow, flux.training_loss)(
            t5_tokens, clip_tokens, x, guidance
        )

    @partial(mx.compile, inputs=state, outputs=state)
    def compute_loss_and_accumulate_grads(
        t5_tokens, clip_tokens, x, guidance, prev_grads
    ):
        loss, grads = nn.value_and_grad(flux.flow, flux.training_loss)(
            t5_tokens, clip_tokens, x, guidance
        )
        grads = tree_map(lambda a, b: a + b, prev_grads, grads)
        return loss, grads

    @partial(mx.compile, inputs=state, outputs=state)
    def grad_accumulate_and_step(t5_tokens, clip_tokens, x, guidance, prev_grads):
        loss, grads = nn.value_and_grad(flux.flow, flux.training_loss)(
            t5_tokens, clip_tokens, x, guidance
        )
        grads = tree_map(lambda a, b: a + b, prev_grads, grads)
        grads = average_gradients(grads)
        optimizer.update(flux.flow, grads)

        return loss

    def step(t5_tokens, clip_tokens, x, guidance, prev_grads, perform_step):
        if prev_grads is None:
            if perform_step:
                return single_step(t5_tokens, clip_tokens, x, guidance), None
            else:
                return compute_loss_and_grads(t5_tokens, clip_tokens, x, guidance)
        else:
            if perform_step:
                return (
                    grad_accumulate_and_step(
                        t5_tokens, clip_tokens, x, guidance, prev_grads
                    ),
                    None,
                )
            else:
                return compute_loss_and_accumulate_grads(
                    t5_tokens, clip_tokens, x, guidance, prev_grads
                )

    print("Encoding training images to latent space")
    x = extract_latent_vectors(flux, args.image_folder)
    t5_tokens, clip_tokens = flux.tokenize([args.prompt] * len(x))
    guidance = mx.full((args.batch_size,), 4.0, dtype=flux.dtype)

    # An initial generation to compare
    generate_progress_images(0, flux, args)

    grads = None
    losses = []
    tic = time.time()
    batches = iterate_batches(t5_tokens, clip_tokens, x, args.batch_size)
    for i, batch in zip(range(args.iterations), batches):
        loss, grads = step(*batch, guidance, grads, (i + 1) % args.grad_accumulate == 0)
        mx.eval(loss, grads, state)
        losses.append(loss.item())

        if (i + 1) % 10 == 0:
            toc = time.time()
            peak_mem = mx.metal.get_peak_memory() / 1024**3
            print(
                f"Iter: {i+1} Loss: {sum(losses) / 10:.3f} "
                f"It/s: {10 / (toc - tic):.3f} "
                f"Peak mem: {peak_mem:.3f} GB"
            )

        if (i + 1) % args.progress_every == 0:
            generate_progress_images(i + 1, flux, args)

        if (i + 1) % args.checkpoint_every == 0:
            pass
            # save_checkpoints(i + 1, sd, args)

        if (i + 1) % 10 == 0:
            losses = []
            tic = time.time()
