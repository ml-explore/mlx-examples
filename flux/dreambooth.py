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
from mlx.utils import tree_map, tree_reduce, tree_unflatten
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

    def generate_prior_preservation(self):
        """Generate some images and mix them with the training images to avoid
        overfitting to the dataset."""

        prior_preservation = self.index.get("prior_preservation", None)
        if not prior_preservation:
            return

        # Select a random set of prompts from the available ones
        prior_prompts = mx.random.randint(
            low=0,
            high=len(prior_preservation["prompts"]),
            shape=(prior_preservation["n_images"],),
        ).tolist()

        # For each prompt
        for prompt_idx in tqdm(prior_prompts):
            # Create the generator
            latents = self.flux.generate_latents(
                prior_preservation["prompts"][prompt_idx],
                num_steps=prior_preservation.get(
                    "num_steps", 2 if "schnell" in self.flux.name else 35
                ),
            )

            # Extract the t5 and clip features
            conditioning = next(latents)
            mx.eval(conditioning)
            t5_feat = conditioning[2]
            clip_feat = conditioning[4]
            del conditioning

            # Do the denoising
            for x_t in latents:
                mx.eval(x_t)

            # Append everything in the data lists
            self.latents.append(x_t)
            self.t5_features.append(t5_feat)
            self.clip_features.append(clip_feat)

    def iterate(self, batch_size):
        while True:
            indices = mx.random.randint(0, len(self.latents), (batch_size,)).tolist()
            x = mx.concatenate([self.latents[i] for i in indices])
            t5 = mx.concatenate([self.t5_features[i] for i in indices])
            clip = mx.concatenate([self.clip_features[i] for i in indices])
            mx.eval(x, t5, clip)
            yield x, t5, clip


def linear_to_lora_layers(flux, args):
    lora_layers = []
    rank = args.lora_rank
    for name, mod in flux.flow.named_modules():
        if ".img_attn" not in name and ".txt_attn" not in name:
            continue
        if ".qkv" in name or ".proj" in name:
            lora_layers.append((name, LoRALinear.from_base(mod, r=rank)))
    flux.flow.update_modules(tree_unflatten(lora_layers))


def decode_latents(flux, x):
    decoded = []
    for i in tqdm(range(len(x))):
        decoded.append(flux.decode(x[i : i + 1]))
        mx.eval(decoded[-1])
    return mx.concatenate(decoded, axis=0)


def generate_latents(flux, n_images, prompt, steps, seed=None, leave=True):
    with random_state(seed):
        latents = flux.generate_latents(
            prompt,
            n_images=n_images,
            num_steps=steps,
        )
        for x_t in tqdm(latents, total=args.progress_steps, leave=leave):
            mx.eval(x_t)

        return x_t


def generate_progress_images(iteration, flux, args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"out_{iteration:03d}.png"
    print(f"Generating {str(out_file)}", flush=True)
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
        default=1000,
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
        "--lora-rank", type=int, default=32, help="LoRA rank for finetuning"
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=100, help="Learning rate warmup"
    )
    parser.add_argument(
        "--learning-rate", type=float, default="1e-5", help="Learning rate for training"
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

    parser.add_argument("dataset")

    args = parser.parse_args()

    # Initialize the seed but different per worker if we are in a distributed
    # setting.
    mx.random.seed(0xF0F0F0F0 + mx.distributed.init().rank())

    flux = FluxPipeline("flux-" + args.model)
    flux.ensure_models_are_loaded()
    flux.flow.freeze()
    with random_state(0x0F0F0F0F):
        linear_to_lora_layers(flux, args)

    trainable_params = tree_reduce(
        lambda acc, x: acc + x.size, flux.flow.trainable_parameters(), 0
    )
    print(f"Training {trainable_params / 1024**2:.3f}M parameters", flush=True)

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
    def compute_loss_and_grads(t5_feat, clip_feat, x, guidance):
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
                        x, t5_feat, clip_feat, x, guidance, prev_grads
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
    dataset.generate_prior_preservation()
    guidance = mx.full((args.batch_size,), 4.0, dtype=flux.dtype)

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
            pass
            # save_checkpoints(i + 1, sd, args)

        if (i + 1) % 10 == 0:
            losses = []
            tic = time.time()
