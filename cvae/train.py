import argparse
import math
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import dataset
import loss
import model
import utils


def loss_fn(model, X):
    # Reconstruct the input images
    X_recon = model(X)
    # Compute the loss between the input images and the reconstructed images
    return loss.mse_kl_loss(X, X_recon, model.get_kl_div())


def save_model(vae, name, epoch):
    fname = f"{name}_{epoch:03d}.npz"
    utils.ensure_folder_exists(fname)
    vae.save(fname)


def train_epoch(model, tr_iter, loss_and_grad_fn, optimizer, epoch):
    # Set model to training mode
    model.train()

    # Reset stats
    loss_acc = mx.array(0.0)
    throughput_acc = mx.array(0.0)

    # Iterate over training batches
    for batch_count, batch in enumerate(tr_iter):
        X = mx.array(batch["image"])

        throughput_tic = time.perf_counter()

        # Forward pass + backward pass + update
        loss, grads = loss_and_grad_fn(model, X)
        optimizer.update(model, grads)

        # Evaluate updated model parameters
        mx.eval(model.parameters(), optimizer.state)

        throughput_toc = time.perf_counter()
        throughput = X.shape[0] / (throughput_toc - throughput_tic)
        throughput_acc += throughput
        loss_acc += loss

        if batch_count > 0 and (batch_count % 10 == 0):
            print(
                " | ".join(
                    [
                        f"Epoch {epoch:4d}",
                        f"Loss {loss.item():10.2f}",
                        f"Throughput {throughput:8.2f} im/s",
                        f"Batch {batch_count:5d}",
                    ]
                ),
                end="\r",
            )

        batch_count = batch_count + 1

        #### end of loop over training batches ####

    return loss_acc, throughput_acc, batch_count


def train(batch_size, num_epochs, learning_rate, num_latent_dims, max_num_filters):
    # Load the data
    img_size = (64, 64)
    tr_iter, _, num_img_channels = dataset.mnist(
        batch_size=batch_size, img_size=img_size
    )

    # Load the model
    vae = model.CVAE(num_latent_dims, num_img_channels, max_num_filters)

    # Allocate memory and initialize parameters
    mx.eval(vae.parameters())
    print("Number of trainable params: {:0.04f} M".format(vae.num_params() / 1e6))

    # Loss and optimizer
    loss_and_grad_fn = nn.value_and_grad(vae, loss_fn)
    optimizer = optim.AdamW(learning_rate=learning_rate)

    # File name to save model every epoch
    fname_save_every_epoch = f"models/vae_mnist_filters_{vae.max_num_filters:04d}_dims_{vae.num_latent_dims:04d}"

    for e in range(num_epochs):
        # Reset iterators and stats at the beginning of each epoch
        tr_iter.reset()

        # Train one epoch
        tic = time.perf_counter()
        loss_acc, throughput_acc, batch_count = train_epoch(
            vae, tr_iter, loss_and_grad_fn, optimizer, e
        )
        toc = time.perf_counter()

        # Print stats
        print(
            " | ".join(
                [
                    f"Epoch {e:4d}",
                    f"Loss {(loss_acc.item() / batch_count):10.2f}",
                    f"Throughput {(throughput_acc.item() / batch_count):8.2f} im/s",
                    f"Time {toc - tic:8.1f} (s)",
                ]
            )
        )

        # Save model in every epoch
        save_model(vae, fname_save_every_epoch, e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU acceleration",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--batchsize", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--max_filters",
        type=int,
        default=64,
        help="Maximum number of filters in the convolutional layers",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    parser.add_argument(
        "--latent_dims",
        type=int,
        default=8,
        help="Number of latent dimensions (positive integer)",
    )

    args = parser.parse_args()

    if args.cpu:
        mx.set_default_device(mx.cpu)

    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    print("Options: ")
    print(f"  Device: {'GPU' if not args.cpu else 'CPU'}")
    print(f"  Seed: {args.seed}")
    print(f"  Batch size: {args.batchsize}")
    print(f"  Max number of filters: {args.max_filters}")
    print(f"  Number of epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Number of latent dimensions: {args.latent_dims}")

    train(args.batchsize, args.epochs, args.lr, args.latent_dims, args.max_filters)
