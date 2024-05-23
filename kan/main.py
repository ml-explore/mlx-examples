import argparse
import time
from functools import partial

# Train on MNIST
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from kan import KAN

import mnist
from tqdm import tqdm


def loss_fn(model, X, y):
    return nn.losses.cross_entropy(model(X), y, reduction="mean")

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

    # Define model
    model = KAN([28 * 28, 64, 10])
    mx.eval(model.parameters())

    # Define optimizer
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=1e-4)
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
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist"],
        help="The dataset to use.",
    )
    args = parser.parse_args()
    main(args)
