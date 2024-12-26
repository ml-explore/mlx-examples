# Copyright © 2024 Apple Inc.

import argparse
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten, tree_map, tree_reduce
from PIL import Image

from flux import FluxPipeline, Trainer, load_dataset, save_config


def generate_progress_images(iteration, flux, args):
    """Generate images to monitor the progress of the finetuning."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{iteration:07d}_progress.png"
    print(f"Generating {str(out_file)}", flush=True)

    # Generate some images and arrange them in a grid
    n_rows = 2
    n_images = 4
    x = flux.generate_images(
        args.progress_prompt,
        n_images,
        args.progress_steps,
    )
    x = mx.pad(x, [(0, 0), (4, 4), (4, 4), (0, 0)])
    B, H, W, C = x.shape
    x = x.reshape(n_rows, B // n_rows, H, W, C).transpose(0, 2, 1, 3, 4)
    x = x.reshape(n_rows * H, B // n_rows * W, C)
    x = mx.pad(x, [(4, 4), (4, 4), (0, 0)])
    x = (x * 255).astype(mx.uint8)

    # Save them to disc
    im = Image.fromarray(np.array(x))
    im.save(out_file)


def save_adapters(adapter_name, flux, args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / adapter_name
    print(f"Saving {str(out_file)}")

    mx.save_safetensors(
        str(out_file),
        dict(tree_flatten(flux.flow.trainable_parameters())),
        metadata={
            "lora_rank": str(args.lora_rank),
            "lora_blocks": str(args.lora_blocks),
        },
    )


def setup_arg_parser():
    """Set up and return the argument parser."""
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
        "--guidance", type=float, default=4.0, help="The guidance factor to use."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=600,
        help="How many iterations to train for",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="The batch size to use when training the stable diffusion model",
    )
    parser.add_argument(
        "--resolution",
        type=lambda x: tuple(map(int, x.split("x"))),
        default=(512, 512),
        help="The resolution of the training images",
    )
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=5,
        help="Augment the images by random cropping and panning",
    )
    parser.add_argument(
        "--progress-prompt",
        required=True,
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
        "--lora-blocks",
        type=int,
        default=-1,
        help="Train the last LORA_BLOCKS transformer blocks",
    )
    parser.add_argument(
        "--lora-rank", type=int, default=8, help="LoRA rank for finetuning"
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=100, help="Learning rate warmup"
    )
    parser.add_argument(
        "--learning-rate", type=float, default="1e-4", help="Learning rate for training"
    )
    parser.add_argument(
        "--grad-accumulate",
        type=int,
        default=4,
        help="Accumulate gradients for that many iterations before applying them",
    )
    parser.add_argument(
        "--output-dir", default="mlx_output", help="Folder to save the checkpoints in"
    )

    parser.add_argument("dataset")
    return parser


if __name__ == "__main__":
    parser = setup_arg_parser()
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_config(vars(args), output_path / "adapter_config.json")

    # Load the model and set it up for LoRA training. We use the same random
    # state when creating the LoRA layers so all workers will have the same
    # initial weights.
    mx.random.seed(0x0F0F0F0F)
    flux = FluxPipeline("flux-" + args.model)
    flux.flow.freeze()
    flux.linear_to_lora_layers(args.lora_rank, args.lora_blocks)

    # Reset the seed to a different seed per worker if we are in distributed
    # mode so that each worker is working on different data, diffusion step and
    # random noise.
    mx.random.seed(0xF0F0F0F0 + mx.distributed.init().rank())

    # Report how many parameters we are training
    trainable_params = tree_reduce(
        lambda acc, x: acc + x.size, flux.flow.trainable_parameters(), 0
    )
    print(f"Training {trainable_params / 1024 ** 2:.3f}M parameters", flush=True)

    # Set up the optimizer and training steps. The steps are a bit verbose to
    # support gradient accumulation together with compilation.
    warmup = optim.linear_schedule(0, args.learning_rate, args.warmup_steps)
    cosine = optim.cosine_decay(
        args.learning_rate, args.iterations // args.grad_accumulate
    )
    lr_schedule = optim.join_schedules([warmup, cosine], [args.warmup_steps])
    optimizer = optim.Adam(learning_rate=lr_schedule)
    state = [flux.flow.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def single_step(x, t5_feat, clip_feat, guidance):
        loss, grads = nn.value_and_grad(flux.flow, flux.training_loss)(
            x, t5_feat, clip_feat, guidance
        )
        grads = average_gradients(grads)
        optimizer.update(flux.flow, grads)

        return loss

    @partial(mx.compile, inputs=state, outputs=state)
    def compute_loss_and_grads(x, t5_feat, clip_feat, guidance):
        return nn.value_and_grad(flux.flow, flux.training_loss)(
            x, t5_feat, clip_feat, guidance
        )

    @partial(mx.compile, inputs=state, outputs=state)
    def compute_loss_and_accumulate_grads(x, t5_feat, clip_feat, guidance, prev_grads):
        loss, grads = nn.value_and_grad(flux.flow, flux.training_loss)(
            x, t5_feat, clip_feat, guidance
        )
        grads = tree_map(lambda a, b: a + b, prev_grads, grads)
        return loss, grads

    @partial(mx.compile, inputs=state, outputs=state)
    def grad_accumulate_and_step(x, t5_feat, clip_feat, guidance, prev_grads):
        loss, grads = nn.value_and_grad(flux.flow, flux.training_loss)(
            x, t5_feat, clip_feat, guidance
        )
        grads = tree_map(
            lambda a, b: (a + b) / args.grad_accumulate,
            prev_grads,
            grads,
        )
        grads = average_gradients(grads)
        optimizer.update(flux.flow, grads)

        return loss

    # We simply route to the appropriate step based on whether we have
    # gradients from a previous step and whether we should be performing an
    # update or simply computing and accumulating gradients in this step.
    def step(x, t5_feat, clip_feat, guidance, prev_grads, perform_step):
        if prev_grads is None:
            if perform_step:
                return single_step(x, t5_feat, clip_feat, guidance), None
            else:
                return compute_loss_and_grads(x, t5_feat, clip_feat, guidance)
        else:
            if perform_step:
                return (
                    grad_accumulate_and_step(
                        x, t5_feat, clip_feat, guidance, prev_grads
                    ),
                    None,
                )
            else:
                return compute_loss_and_accumulate_grads(
                    x, t5_feat, clip_feat, guidance, prev_grads
                )

    dataset = load_dataset(args.dataset)
    trainer = Trainer(flux, dataset, args)
    trainer.encode_dataset()

    guidance = mx.full((args.batch_size,), args.guidance, dtype=flux.dtype)

    # An initial generation to compare
    generate_progress_images(0, flux, args)

    grads = None
    losses = []
    tic = time.time()
    for i, batch in zip(range(args.iterations), trainer.iterate(args.batch_size)):
        loss, grads = step(*batch, guidance, grads, (i + 1) % args.grad_accumulate == 0)
        mx.eval(loss, grads, state)
        losses.append(loss.item())

        if (i + 1) % 10 == 0:
            toc = time.time()
            peak_mem = mx.metal.get_peak_memory() / 1024**3
            print(
                f"Iter: {i + 1} Loss: {sum(losses) / 10:.3f} "
                f"It/s: {10 / (toc - tic):.3f} "
                f"Peak mem: {peak_mem:.3f} GB",
                flush=True,
            )

        if (i + 1) % args.progress_every == 0:
            generate_progress_images(i + 1, flux, args)

        if (i + 1) % args.checkpoint_every == 0:
            save_adapters(f"{i + 1:07d}_adapters.safetensors", flux, args)

        if (i + 1) % 10 == 0:
            losses = []
            tic = time.time()

    save_adapters("final_adapters.safetensors", flux, args)
    print("Training successful.")
