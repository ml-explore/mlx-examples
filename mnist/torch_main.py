# Copyright © 2023 Apple Inc.

import argparse
import time

import torch

import mnist


class MLP(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super().__init__()
        layer_sizes = [hidden_dim] * num_layers
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(idim, odim)
                for idim, odim in zip(
                    [input_dim] + layer_sizes, layer_sizes + [output_dim]
                )
            ]
        )

    def forward(self, x):
        x = self.layers[0](x)
        for l in self.layers[1:]:
            x = l(x.relu())
        return x


def loss_fn(model, X, y):
    logits = model(X)
    return torch.nn.functional.cross_entropy(logits, y)


@torch.no_grad()
def eval_fn(model, X, y):
    logits = model(X)
    return torch.mean((logits.argmax(-1) == y).float())


def batch_iterate(batch_size, X, y, device):
    perm = torch.randperm(len(y), device=device)
    for s in range(0, len(y), batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids], y[ids]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a simple MLP on MNIST with PyTorch.")
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
        torch.set_num_threads(1)
        device = "cpu"
    else:
        device = "mps"
    seed = 0
    num_layers = 2
    hidden_dim = 32
    num_classes = 10
    batch_size = 256
    num_epochs = 10
    learning_rate = 1e-1

    # Load the data
    def to_tensor(x):
        if x.dtype != "uint32":
            return torch.from_numpy(x).to(device)
        else:
            return torch.from_numpy(x.astype(int)).to(device)

    train_images, train_labels, test_images, test_labels = map(
        to_tensor, getattr(mnist, args.dataset)()
    )

    # Load the model
    model = MLP(num_layers, train_images.shape[-1], hidden_dim, num_classes).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0)

    for e in range(num_epochs):
        tic = time.perf_counter()
        for X, y in batch_iterate(batch_size, train_images, train_labels, device):
            opt.zero_grad()
            loss_fn(model, X, y).backward()
            opt.step()
        accuracy = eval_fn(model, test_images, test_labels)
        toc = time.perf_counter()
        print(
            f"Epoch {e}: Test accuracy {accuracy.item():.3f},"
            f" Time {toc - tic:.3f} (s)"
        )
