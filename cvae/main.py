# Copyright © 2023-2024 Apple Inc.

import argparse
import time
from functools import partial
from pathlib import Path

import dataset
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import vae
from mlx.utils import tree_flatten
from PIL import Image


def grid_image_from_batch(image_batch, num_rows):
    """
    Generate a grid image from a batch of images.
    Assumes input has shape (B, H, W, C).
    """

    B, H, W, _ = image_batch.shape

    num_cols = B // num_rows

    # Calculate the size of the output grid image
    grid_height = num_rows * H
    grid_width = num_cols * W

    # Normalize and convert to the desired data type
    image_batch = np.array(image_batch * 255).astype(np.uint8)

    # Reshape the batch of images into a 2D grid
    grid_image = image_batch.reshape(num_rows, num_cols, H, W, -1)
    grid_image = grid_image.swapaxes(1, 2)
    grid_image = grid_image.reshape(grid_height, grid_width, -1)

    # Convert the grid to a PIL Image
    return Image.fromarray(grid_image.squeeze())


def loss_fn(model, X):
    X_recon, mu, logvar = model(X)

    # Reconstruction loss
    recon_loss = nn.losses.mse_loss(X_recon, X, reduction="sum")

    # KL divergence between encoder distribution and standard normal:
    kl_div = -0.5 * mx.sum(1 + logvar - mu.square() - logvar.exp())

    # Total loss
    return recon_loss + kl_div


def reconstruct(model, batch, out_file):
    # Reconstruct a single batch only
    images = mx.array(batch["image"])
    images_recon = model(images)[0]
    paired_images = mx.stack([images, images_recon]).swapaxes(0, 1).flatten(0, 1)
    grid_image = grid_image_from_batch(paired_images, num_rows=16)
    grid_image.save(out_file)


def generate(
    model,
    out_file,
    num_samples=128,
):
    # Sample from the latent distribution:
    z = mx.random.normal([num_samples, model.num_latent_dims])

    # Decode the latent vectors to images:
    images = model.decode(z)

    # Save all images in a single file
    grid_image = grid_image_from_batch(images, num_rows=8)
    grid_image.save(out_file)


def main(args):
    # Load the data
    img_size = (64, 64, 1)
    train_iter, test_iter = dataset.mnist(
        batch_size=args.batch_size, img_size=img_size[:2]
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load the model
    model = vae.CVAE(args.latent_dims, img_size, args.max_filters)
    mx.eval(model.parameters())

    num_params = sum(x.size for _, x in tree_flatten(model.trainable_parameters()))
    print("Number of trainable params: {:0.04f} M".format(num_params / 1e6))

    optimizer = optim.AdamW(learning_rate=args.lr)

    # Batches for reconstruction
    train_batch = next(train_iter)
    test_batch = next(test_iter)

    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(X):
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, X)
        optimizer.update(model, grads)
        return loss

    for e in range(1, args.epochs + 1):
        # Reset iterators and stats at the beginning of each epoch
        train_iter.reset()
        model.train()

        # Train one epoch
        tic = time.perf_counter()
        loss_acc = 0.0
        throughput_acc = 0.0

        # Iterate over training batches
        for batch_count, batch in enumerate(train_iter):
            X = mx.array(batch["image"])
            throughput_tic = time.perf_counter()

            # Forward pass + backward pass + update
            loss = step(X)

            # Evaluate updated model parameters
            mx.eval(state)

            throughput_toc = time.perf_counter()
            throughput_acc += X.shape[0] / (throughput_toc - throughput_tic)
            loss_acc += loss.item()

            if batch_count > 0 and (batch_count % 10 == 0):
                print(
                    " | ".join(
                        [
                            f"Epoch {e:4d}",
                            f"Loss {(loss_acc / batch_count):10.2f}",
                            f"Throughput {(throughput_acc / batch_count):8.2f} im/s",
                            f"Batch {batch_count:5d}",
                        ]
                    ),
                    end="\r",
                )
        toc = time.perf_counter()

        print(
            " | ".join(
                [
                    f"Epoch {e:4d}",
                    f"Loss {(loss_acc / batch_count):10.2f}",
                    f"Throughput {(throughput_acc / batch_count):8.2f} im/s",
                    f"Time {toc - tic:8.1f} (s)",
                ]
            )
        )

        model.eval()

        # Reconstruct a batch of training and test images
        reconstruct(model, train_batch, save_dir / f"train_{e:03d}.png")
        reconstruct(model, test_batch, save_dir / f"test_{e:03d}.png")

        # Generate images
        generate(model, save_dir / f"generated_{e:03d}.png")

        model.save_weights(str(save_dir / "weights.npz"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU acceleration",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--max-filters",
        type=int,
        default=64,
        help="Maximum number of filters in the convolutional layers",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    parser.add_argument(
        "--latent-dims",
        type=int,
        default=8,
        help="Number of latent dimensions (positive integer)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models/",
        help="Path to save the model and reconstructed images.",
    )

    args = parser.parse_args()

    if args.cpu:
        mx.set_default_device(mx.cpu)

    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    print("Options: ")
    print(f"  Device: {'GPU' if not args.cpu else 'CPU'}")
    print(f"  Seed: {args.seed}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max number of filters: {args.max_filters}")
    print(f"  Number of epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Number of latent dimensions: {args.latent_dims}")

    main(args)
