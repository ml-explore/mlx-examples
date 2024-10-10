import argparse
import json
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten, tree_map, tree_reduce, tree_unflatten
from PIL import Image
from tqdm import tqdm

from flux import FluxPipeline
from flux.lora import LoRALinear


@contextmanager
def random_state(seed=None):
    s = mx.random.state[0]
    try:
        if seed is not None:
            mx.random.seed(seed)
        yield
    finally:
        mx.random.state[0] = s


class FinetuningDataset:
    def __init__(self, flux, args):
        self.args = args
        self.flux = flux
        self.dataset_base = Path(args.dataset)
        dataset_index = self.dataset_base / "index.json"
        if not dataset_index.exists():
            raise ValueError(f"'{args.dataset}' is not a valid finetuning dataset")
        with open(dataset_index, "r") as f:
            self.index = json.load(f)

        self.latents = []
        self.t5_features = []
        self.clip_features = []

    def encode_images(self):
        """Encode the images in the latent space to prepare for training."""
        self.flux.ae.eval()
        for sample in tqdm(self.index["data"]):
            img = Image.open(self.dataset_base / sample["image"])
            width, height = img.size
            if width != height:
                side = min(width, height)
                img = img.crop(
                    (width - side) / 2,
                    (height - side) / 2,
                    (width + side) / 2,
                    (height + side) / 2,
                )
            img = img.resize(self.args.resolution, Image.LANCZOS)
            img = mx.array(np.array(img))
            img = (img[:, :, :3].astype(flux.dtype) / 255) * 2 - 1
            x_0 = self.flux.ae.encode(img[None])
            x_0 = x_0.astype(flux.dtype)
            mx.eval(x_0)
            self.latents.append(x_0)

    def encode_prompts(self):
        """Pre-encode the prompts so that we don't recompute them during
        training (doesn't allow finetuning the text encoders)."""
        for sample in tqdm(self.index["data"]):
            t5_tok, clip_tok = self.flux.tokenize([sample["text"]])
            t5_feat = self.flux.t5(t5_tok)
            clip_feat = self.flux.clip(clip_tok).pooled_output
            mx.eval(t5_feat, clip_feat)
            self.t5_features.append(t5_feat)
            self.clip_features.append(clip_feat)

    def iterate(self, batch_size):
        xs = mx.concatenate(self.latents)
        t5 = mx.concatenate(self.t5_features)
        clip = mx.concatenate(self.clip_features)
        mx.eval(xs, t5, clip)
        while True:
            indices = mx.random.randint(0, len(self.latents), (batch_size,))
            yield xs[indices], t5[indices], clip[indices]


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


def save_adapters(iteration, flux, args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{iteration:07d}_adapters.safetensors"
    print(f"Saving {str(out_file)}")

    mx.save_safetensors(
        str(out_file),
        dict(tree_flatten(flux.flow.trainable_parameters())),
        metadata={
            "lora_rank": str(args.lora_rank),
            "lora_blocks": str(args.lora_blocks),
        },
    )


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
        "--guidance", type=float, default=4.0, help="The guidance factor to use."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=600,
        help="How many iterations to train for",
    )
    parser.add_argument(
        "--batch_size",
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

    args = parser.parse_args()

    # Initialize the seed but different per worker if we are in a distributed
    # setting.
    mx.random.seed(0xF0F0F0F0 + mx.distributed.init().rank())

    # Load the model and set it up for LoRA training. We use the same random
    # state when creating the LoRA layers so all workers will have the same
    # initial weights.
    flux = FluxPipeline("flux-" + args.model)
    flux.flow.freeze()
    with random_state(0x0F0F0F0F):
        flux.linear_to_lora_layers(args.lora_rank, args.lora_blocks)

    # Report how many parameters we are training
    trainable_params = tree_reduce(
        lambda acc, x: acc + x.size, flux.flow.trainable_parameters(), 0
    )
    print(f"Training {trainable_params / 1024**2:.3f}M parameters", flush=True)

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
        grads = tree_map(lambda a, b: a + b, prev_grads, grads)
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

    print("Create the training dataset.", flush=True)
    dataset = FinetuningDataset(flux, args)
    dataset.encode_images()
    dataset.encode_prompts()
    guidance = mx.full((args.batch_size,), args.guidance, dtype=flux.dtype)

    # An initial generation to compare
    generate_progress_images(0, flux, args)

    grads = None
    losses = []
    tic = time.time()
    for i, batch in zip(range(args.iterations), dataset.iterate(args.batch_size)):
        loss, grads = step(*batch, guidance, grads, (i + 1) % args.grad_accumulate == 0)
        mx.eval(loss, grads, state)
        losses.append(loss.item())

        if (i + 1) % 10 == 0:
            toc = time.time()
            peak_mem = mx.metal.get_peak_memory() / 1024**3
            print(
                f"Iter: {i+1} Loss: {sum(losses) / 10:.3f} "
                f"It/s: {10 / (toc - tic):.3f} "
                f"Peak mem: {peak_mem:.3f} GB",
                flush=True,
            )

        if (i + 1) % args.progress_every == 0:
            generate_progress_images(i + 1, flux, args)

        if (i + 1) % args.checkpoint_every == 0:
            save_adapters(i + 1, flux, args)

        if (i + 1) % 10 == 0:
            losses = []
            tic = time.time()
