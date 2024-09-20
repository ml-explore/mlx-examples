# Copyright © 2024 Apple Inc.

import argparse
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from PIL import Image
from tqdm import tqdm

from stable_diffusion import StableDiffusion


def extract_latent_vectors(sd, image_folder):
    latents = []
    for image in tqdm(Path(image_folder).iterdir()):
        img = Image.open(image)
        img = mx.array(np.array(img))
        img = (img[:, :, :3].astype(mx.float32) / 255) * 2 - 1
        x_0, _ = sd.autoencoder.encode(img[None])
        mx.eval(x_0)
        latents.append(x_0)
    return mx.concatenate(latents)


def generate_latents(sd, n_images, prompt, steps, cfg_weight, seed=None, leave=True):
    latents = sd.generate_latents(
        prompt,
        n_images=n_images,
        cfg_weight=cfg_weight,
        num_steps=steps,
        seed=seed,
        negative_text="",
    )
    for x_t in tqdm(latents, total=args.progress_steps, leave=leave):
        mx.eval(x_t)

    return x_t


def decode_latents(sd, x):
    decoded = []
    for i in tqdm(range(len(x))):
        decoded.append(sd.decode(x[i : i + 1]))
        mx.eval(decoded[-1])
    return mx.concatenate(decoded, axis=0)


def generate_progress_images(iteration, sd, args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"out_{iteration:03d}.png"
    print(f"Generating {str(out_file)}")
    # Generate the latent vectors using diffusion
    n_images = 4
    latents = generate_latents(
        sd,
        n_images,
        args.progress_prompt,
        args.progress_steps,
        args.progress_cfg,
        seed=42,
    )

    # Arrange them on a grid
    n_rows = 2
    x = decode_latents(sd, latents)
    x = mx.pad(x, [(0, 0), (8, 8), (8, 8), (0, 0)])
    B, H, W, C = x.shape
    x = x.reshape(n_rows, B // n_rows, H, W, C).transpose(0, 2, 1, 3, 4)
    x = x.reshape(n_rows * H, B // n_rows * W, C)
    x = (x * 255).astype(mx.uint8)

    # Save them to disc
    im = Image.fromarray(np.array(x))
    im.save(out_file)


def save_checkpoints(iteration, sd, args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    unet_file = str(out_dir / f"unet_{iteration:03d}.safetensors")
    print(f"Saving {unet_file}")
    sd.unet.save_weights(unet_file)
    if args.train_text_encoder:
        te_file = str(out_dir / f"text_encoder_{iteration:03d}.safetensors")
        print(f"Saving {te_file}")
        sd.text_encoder.save_weights(te_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finetune SD to generate images with a specific subject"
    )

    parser.add_argument(
        "--model",
        default="CompVis/stable-diffusion-v1-4",
        choices=[
            "stabilityai/stable-diffusion-2-1-base",
            "CompVis/stable-diffusion-v1-4",
        ],
        help="Which stable diffusion model to train",
    )
    parser.add_argument(
        "--train-text-encoder",
        action="store_true",
        help="Train the text encoder as well as the UNet",
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
        default=4,
        help="The batch size to use when training the stable diffusion model",
    )
    parser.add_argument(
        "--progress-prompt",
        help="Use this prompt when generating images for evaluation",
    )
    parser.add_argument(
        "--progress-cfg",
        type=float,
        default=7.5,
        help="Use this classifier free guidance weight when generating images for evaluation",
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
        "--learning_rate", type=float, default="1e-6", help="Learning rate for training"
    )
    parser.add_argument(
        "--predict-x0",
        action="store_false",
        dest="predict_noise",
        help="Compute the loss on x0 instead of the noise",
    )
    parser.add_argument(
        "--prior-preservation-weight",
        type=float,
        default=0,
        help="The loss weight for the prior preservation batches",
    )
    parser.add_argument(
        "--prior-preservation-images",
        type=int,
        default=100,
        help="How many prior preservation images to use",
    )
    parser.add_argument(
        "--prior-preservation-prompt", help="The prompt to use for prior preservation"
    )
    parser.add_argument(
        "--prior-preservation-steps",
        default=50,
        type=int,
        help="How many steps to use to generate prior images",
    )
    parser.add_argument(
        "--prior-preservation-cfg",
        default=7.5,
        type=float,
        help="The CFG weight to use to generate prior images",
    )
    parser.add_argument(
        "--output-dir", default="mlx_output", help="Folder to save the checkpoints in"
    )

    parser.add_argument("prompt")
    parser.add_argument("image_folder")

    args = parser.parse_args()

    args.progress_prompt = args.progress_prompt or args.prompt
    args.prior_preservation_prompt = args.prior_preservation_prompt or args.prompt

    sd = StableDiffusion(args.model)
    sd.ensure_models_are_loaded()

    optimizer = optim.Adam(learning_rate=1e-6)

    def loss_fn(params, text, x, weights):
        sd.unet.update(params["unet"])
        if "text_encoder" in params:
            sd.text_encoder.update(params["text_encoder"])
        loss = sd.training_loss(text, x, pred_noise=args.predict_noise)
        loss = loss * weights
        return loss.mean()

    state = [sd.unet.state, optimizer.state, mx.random.state]
    if args.train_text_encoder:
        state.append(sd.text_encoder.state)

    @partial(mx.compile, inputs=state, outputs=state)
    def step(text, x, prior_text=None, prior_x=None, prior_weight=None):
        # Get the parameters we are calculating gradients for
        params = {"unet": sd.unet.trainable_parameters()}
        if args.train_text_encoder:
            params["text_encoder"] = sd.text_encoder.trainable_parameters()

        # Combine the prior preservation if needed
        if prior_weight is None:
            weights = mx.ones(len(x))
        else:
            weights = mx.array([1] * len(x) + [prior_weight] * len(prior_x))
            x = mx.concatenate([x, prior_x])
            text = mx.concatenate([text, prior_text])

        # Calculate the loss and new parameters
        loss, grads = mx.value_and_grad(loss_fn)(params, text, x, weights)
        params = optimizer.apply_gradients(grads, params)

        # Update the models
        sd.unet.update(params["unet"])
        if "text_encoder" in params:
            sd.text_encoder.update(params["text_encoder"])

        return loss

    print("Encoding training images to latent space")
    x = extract_latent_vectors(sd, args.image_folder)
    text = sd._tokenize(sd.tokenizer, args.prompt, None)
    text = mx.repeat(text, len(x), axis=0)
    prior_x = None
    prior_text = None

    if args.prior_preservation_weight > 0:
        print("Generating prior images")
        batch_size = 4
        prior_x = mx.zeros(
            (
                batch_size
                * (args.prior_preservation_images + batch_size - 1)
                // batch_size,
                *x.shape[1:],
            ),
            dtype=x.dtype,
        )
        prior_text = sd._tokenize(sd.tokenizer, args.prior_preservation_prompt, None)
        prior_text = mx.repeat(prior_text, len(prior_x), axis=0)
        for i in tqdm(range(0, args.prior_preservation_images, batch_size)):
            prior_batch = generate_latents(
                sd,
                batch_size,
                args.prior_preservation_prompt,
                args.prior_preservation_steps,
                args.prior_preservation_cfg,
                leave=False,
            )
            prior_x[i : i + batch_size] = prior_batch
            mx.async_eval(prior_x)

    # An initial generation to compare
    generate_progress_images(0, sd, args)

    losses = []
    tic = time.time()
    for i in range(args.iterations):
        indices = (mx.random.uniform(shape=(args.batch_size,)) * len(x)).astype(
            mx.uint32
        )
        if args.prior_preservation_weight > 0.0:
            prior_indices = (
                mx.random.uniform(shape=(args.batch_size,)) * len(prior_x)
            ).astype(mx.uint32)
            loss = step(
                text[indices],
                x[indices],
                prior_text[prior_indices],
                prior_x[prior_indices],
                args.prior_preservation_weight,
            )
        else:
            loss = step(text[indices], x[indices])
        mx.eval(loss, state)
        losses.append(loss.item())

        if (i + 1) % 10 == 0:
            toc = time.time()
            print(
                f"Iter: {i+1} Loss: {sum(losses) / 10:.3f} "
                f"It/s: {10 / (toc - tic):.3f}"
            )

        if (i + 1) % args.progress_every == 0:
            generate_progress_images(i + 1, sd, args)

        if (i + 1) % args.checkpoint_every == 0:
            save_checkpoints(i + 1, sd, args)

        if (i + 1) % 10 == 0:
            losses = []
            tic = time.time()
