# Copyright © 2024 Apple Inc.

import argparse
import json
import time
from functools import partial
from pathlib import Path
import os

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten, tree_map, tree_reduce
from PIL import Image
from tqdm import tqdm

from huggingface_hub import HfApi, interpreter_login
from huggingface_hub.utils import HfFolder

from flux import FluxPipeline


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

    def _random_crop_resize(self, img):
        resolution = self.args.resolution
        width, height = img.size

        a, b, c, d = mx.random.uniform(shape=(4,), stream=mx.cpu).tolist()

        # Random crop the input image between 0.8 to 1.0 of its original dimensions
        crop_size = (
            max((0.8 + 0.2 * a) * width, resolution[0]),
            max((0.8 + 0.2 * a) * height, resolution[1]),
        )
        pan = (width - crop_size[0], height - crop_size[1])
        img = img.crop(
            (
                pan[0] * b,
                pan[1] * c,
                crop_size[0] + pan[0] * b,
                crop_size[1] + pan[1] * c,
            )
        )

        # Fit the largest rectangle with the ratio of resolution in the image
        # rectangle.
        width, height = crop_size
        ratio = resolution[0] / resolution[1]
        r1 = (height * ratio, height)
        r2 = (width, width / ratio)
        r = r1 if r1[0] <= width else r2
        img = img.crop(
            (
                (width - r[0]) / 2,
                (height - r[1]) / 2,
                (width + r[0]) / 2,
                (height + r[1]) / 2,
            )
        )

        # Finally resize the image to resolution
        img = img.resize(resolution, Image.LANCZOS)

        return mx.array(np.array(img))

    def encode_images(self):
        """Encode the images in the latent space to prepare for training."""
        self.flux.ae.eval()
        for sample in tqdm(self.index["data"]):
            input_img = Image.open(self.dataset_base / sample["image"])
            for i in range(self.args.num_augmentations):
                img = self._random_crop_resize(input_img)
                img = (img[:, :, :3].astype(self.flux.dtype) / 255) * 2 - 1
                x_0 = self.flux.ae.encode(img[None])
                x_0 = x_0.astype(self.flux.dtype)
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
        n_aug = self.args.num_augmentations
        while True:
            x_indices = mx.random.permutation(len(self.latents))
            c_indices = x_indices // n_aug
            for i in range(0, len(self.latents), batch_size):
                x_i = x_indices[i : i + batch_size]
                c_i = c_indices[i : i + batch_size]
                yield xs[x_i], t5[c_i], clip[c_i]


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

def push_to_hub(args):
    if args.hf_token is None:
        interpreter_login(new_session=False, write_permission=True)
    else:
        HfFolder.save_token(args.hf_token)

    repo_id = args.hf_repo_id or f"{HfFolder.get_token_username()}/{args.output_dir}"
    
    readme_content = generate_readme(args, repo_id)
    readme_path = os.path.join(args.output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    api = HfApi()

    api.create_repo(
        repo_id,
        private=args.hf_private,
        exist_ok=True
    )

    api.upload_folder(
        repo_id=repo_id,
        folder_path=args.output_dir,
        ignore_patterns=["*.yaml", "*.pt"],
        repo_type="model",
    )

def generate_readme(args, repo_id):
    import yaml
    import re
    base_model = f"flux-{args.model}"
    tags = [
        "text-to-image",
        "flux",
        "lora",
        "diffusers",
        "template:sd-lora",
        "mlx",
        "mlx-trainer"
    ]

    widgets = []
    sample_image_paths = []
    # Look for progress images directly in the output directory
    for filename in os.listdir(args.output_dir):
        match = re.search(r"(\d+)_progress\.png$", filename)
        if match:
            iteration = int(match.group(1))
            sample_image_paths.append((iteration, filename))

    sample_image_paths.sort(key=lambda x: x[0], reverse=True)
    
    if sample_image_paths:
        widgets.append(
            {
                "text": args.progress_prompt,
                "output": {
                    "url": sample_image_paths[0][1]
                },
            }
        )

    readme_content = f"""---
tags:
{yaml.dump(tags, indent=4).strip()}
{"widget:" if sample_image_paths else ""}
{yaml.dump(widgets, indent=4).strip() if widgets else ""}
base_model: {base_model}
license: other
---

# {os.path.basename(args.output_dir)}
Model trained with the MLX Flux Dreambooth script

<Gallery />

## Use it with [MLX](https://github.com/ml-explore/mlx-examples)
```py
from flux import FluxPipeline
import mlx.core as mx
flux = FluxPipeline("flux-{args.model}")
flux.linear_to_lora_layers({args.lora_rank}, {args.lora_blocks})
flux.flow.load_weights("{repo_id}")
image = flux.generate_images("{args.progress_prompt}", n_images=1, num_steps={args.progress_steps})
image.save("my_image.png")
```

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)
```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained('black-forest-labs/{args.model}', torch_dtype=torch.bfloat16).to('cuda')
pipeline.load_lora_weights('{repo_id}')
image = pipeline({args.progress_prompt}').images[0]
image.save("my_image.png")
```

For more details on using Flux, check the [Flux documentation](https://github.com/black-forest-labs/flux).
"""
    return readme_content

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
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the model to Hugging Face Hub after training",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token for pushing to Hub",
    )
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        default=None,
        help="Hugging Face repository ID for pushing to Hub",
    )
    parser.add_argument(
        "--hf_private",
        action="store_true",
        help="Make the Hugging Face repository private",
    )
    parser.add_argument("dataset")

    args = parser.parse_args()

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
    
    if args.push_to_hub:
        push_to_hub(args)
