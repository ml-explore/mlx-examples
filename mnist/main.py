# Copyright © 2023 Apple Inc.

import argparse
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import mnist


class MLP(nn.Module):
    """A simple MLP."""

    def __init__(
        self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int
    ):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [
            nn.Linear(idim, odim)
            for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

    def __call__(self, x):
        for l in self.layers[:-1]:
            x = nn.relu(l(x))
        return self.layers[-1](x)


def loss_fn(model, X, y):
    return nn.losses.cross_entropy(model(X), y, reduction="mean")


def batch_iterate(batch_size, X, y):
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids], y[ids]


def main(args):
    seed = 0
    num_layers = 2
    hidden_dim = 32
    num_classes = 10
    batch_size = 256
    num_epochs = 10
    learning_rate = 1e-1

    np.random.seed(seed)

    # Load the data
    train_images, train_labels, test_images, test_labels = map(
        mx.array, getattr(mnist, args.dataset)()
    )

    # Load the model
    model = MLP(num_layers, train_images.shape[-1], hidden_dim, num_classes)
    mx.eval(model.parameters())

    optimizer = optim.SGD(learning_rate=learning_rate)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    @partial(mx.compile, inputs=model.state, outputs=model.state)
    def step(X, y):
        loss, grads = loss_and_grad_fn(model, X, y)
        optimizer.update(model, grads)
        return loss

    @partial(mx.compile, inputs=model.state)
    def eval_fn(X, y):
        return mx.mean(mx.argmax(model(X), axis=1) == y)

    for e in range(num_epochs):
        tic = time.perf_counter()
        for X, y in batch_iterate(batch_size, train_images, train_labels):
            step(X, y)
            mx.eval(model.state)
        accuracy = eval_fn(test_images, test_labels)
        toc = time.perf_counter()
        print(
            f"Epoch {e}: Test accuracy {accuracy.item():.3f},"
            f" Time {toc - tic:.3f} (s)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a simple MLP on MNIST with MLX.")
    parser.add_argument("--gpu", action="store_true", help="Use the Metal back-end.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist"],
        help="The dataset to use.",
    )
    args = parser.parse_args()
    if not args.gpu:
        mx.set_default_device(mx.cpu)
    main(args)
