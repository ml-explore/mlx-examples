# Copyright © 2023-2024 Apple Inc.

from functools import partial

import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from flows import RealNVP
from sklearn import datasets, preprocessing
from tqdm import trange


def get_moons_dataset(n_samples=100_000, noise=0.06):
    """Get two moons dataset with given noise level."""
    x, _ = datasets.make_moons(n_samples=n_samples, noise=noise)
    scaler = preprocessing.StandardScaler()
    x = scaler.fit_transform(x)
    return x


def main(args):
    x = get_moons_dataset(n_samples=100_000, noise=args.noise)

    model = RealNVP(args.n_transforms, args.d_params, args.d_hidden, args.n_layers)
    mx.eval(model.parameters())

    def loss_fn(model, x):
        return -mx.mean(model(x))

    optimizer = optim.Adam(learning_rate=args.learning_rate)

    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x):
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, x)
        optimizer.update(model, grads)
        return loss

    with trange(args.n_steps) as steps:
        for it in steps:
            idx = np.random.choice(x.shape[0], replace=False, size=args.n_batch)
            loss = step(mx.array(x[idx]))
            mx.eval(state)
            steps.set_postfix(val=loss.item())

    # Plot samples from trained flow

    fig, axs = plt.subplots(1, args.n_transforms + 2, figsize=(26, 4))
    cmap = plt.get_cmap("Blues")
    bins = 100

    # Sample from intermediate flow-transformed distributions
    for n_transforms in range(args.n_transforms + 1):
        x_samples = model.sample((100_000, 2), n_transforms=n_transforms)

        axs[n_transforms].hist2d(x_samples[:, 0], x_samples[:, 1], bins=bins, cmap=cmap)
        axs[n_transforms].set_xlim(-2, 2)
        axs[n_transforms].set_ylim(-2, 2)
        axs[n_transforms].set_title(
            f"{n_transforms} transforms" if n_transforms > 0 else "Base distribution"
        )
        axs[n_transforms].set_xticklabels([])
        axs[n_transforms].set_yticklabels([])

    # Plot original data
    axs[-1].hist2d(x[:, 0], x[:, 1], bins=bins, cmap=cmap)
    axs[-1].set_xlim(-2, 2)
    axs[-1].set_ylim(-2, 2)
    axs[-1].set_title("Original data")
    axs[-1].set_xticklabels([])
    axs[-1].set_yticklabels([])

    plt.tight_layout()
    plt.savefig("samples.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_steps", type=int, default=5_000, help="Number of steps to train"
    )
    parser.add_argument("--n_batch", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--n_transforms", type=int, default=6, help="Number of flow transforms"
    )
    parser.add_argument(
        "--d_params", type=int, default=2, help="Dimensionality of modeled distribution"
    )
    parser.add_argument(
        "--d_hidden",
        type=int,
        default=128,
        help="Hidden dimensionality of coupling conditioner",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=4,
        help="Number of layers in coupling conditioner",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--noise", type=float, default=0.06, help="Noise level in two moons dataset"
    )
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    if args.cpu:
        mx.set_default_device(mx.cpu)

    main(args)
