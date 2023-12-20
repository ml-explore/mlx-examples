# Copyright © 2023 Apple Inc.

import math
import time

import datasets
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, dims: int, num_heads: int):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dims)
        self.transformer = nn.TransformerEncoder(num_layers, dims, num_heads)
        self.out_proj = nn.Linear(dims, vocab_size)

    def __call__(self, x):
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        x = self.embedding(x)
        x = self.transformer(x, mask)
        return self.out_proj(x)

    def loss(self, x, y, reduce=True):
        logits = self(x)
        losses = nn.losses.cross_entropy(logits, y)
        mx.simplify(losses)

        return mx.mean(losses) if reduce else mx.mean(losses, axis=(-1, -2))


def to_samples(context_size, dataset):
    tokens = dataset.size
    window_size = context_size + 1  # include target
    samples = tokens - window_size + 1
    X = np.lib.stride_tricks.as_strided(
        dataset,
        shape=(samples, window_size),
        strides=(dataset.itemsize, dataset.itemsize),
    )
    return X[:, :-1], X[:, 1:]


def iterate_batches(batch_size, context_size, dataset):
    inputs, targets = to_samples(context_size, dataset)
    s = 0
    while True:
        if s == 0:
            # Reset permutation:
            perm = np.random.permutation(inputs.shape[0])
        ids = perm[s : s + batch_size]
        yield inputs[ids], targets[ids]
        s += batch_size
        if s >= inputs.shape[0]:
            s = 0


def main(args):
    batch_size = args.batch_size
    context_size = args.context_size
    steps_per_eval = args.steps_per_eval
    steps_per_report = args.steps_per_report

    # Load vocab and dataset:
    vocab, train, valid, test = datasets.load_dataset(args.dataset)

    # Initialize model:
    model = TransformerLM(len(vocab), args.num_blocks, args.dim, args.num_heads)
    mx.eval(model.parameters())
    nparams = sum(
        x.size for k, x in tree_flatten(model.parameters()) if "embedding" not in k
    )
    print(f"Training a transformer with {nparams / 1024**2:.3f} M parameters")

    optimizer = optim.SGD(learning_rate=args.learning_rate)
    loss_and_grad_fn = nn.value_and_grad(model, model.loss)

    def eval_fn(model, dataset):
        inputs, targets = map(mx.array, to_samples(context_size, dataset))
        loss = 0
        for s in range(0, targets.shape[0], batch_size):
            bx, by = inputs[s : s + batch_size], targets[s : s + batch_size]
            bx, by = map(mx.array, (bx, by))
            losses = model.loss(bx, by, reduce=False)
            loss += mx.sum(losses).item()
        return loss / len(targets)

    train_iterator = iterate_batches(batch_size, context_size, train)
    losses = []
    tic = time.perf_counter()
    for it, (inputs, targets) in zip(range(args.num_iters), train_iterator):
        inputs, targets = map(mx.array, (inputs, targets))
        loss, grads = loss_and_grad_fn(inputs, targets)
        model.update(optimizer.apply_gradients(grads, model))
        mx.simplify(loss, model.parameters())
        mx.eval(loss, model.parameters())
        losses.append(loss.item())
        if (it + 1) % steps_per_report == 0:
            train_loss = np.mean(losses)
            toc = time.perf_counter()
            print(
                f"Iter {it + 1}: Train loss {train_loss:.3f}, "
                f"It/sec {steps_per_report / (toc - tic):.3f}"
            )
            losses = []
            tic = time.perf_counter()
        if (it + 1) % steps_per_eval == 0:
            val_loss = eval_fn(model, valid)
            toc = time.perf_counter()
            print(
                f"Iter {it + 1}: "
                f"Val loss {val_loss:.3f}, "
                f"Val ppl {math.exp(val_loss):.3f}, "
                f"Val took {(toc - tic):.3f}s, "
            )
            tic = time.perf_counter()

    if args.eval_test:
        test_loss = eval_fn(model, test)
        test_ppl = math.exp(test_loss)
        print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Train a decoder-only Transformer LM with MLX.")
    parser.add_argument("--gpu", action="store_true", help="Use the Metal back-end.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the RNGs.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ptb",
        choices=["ptb", "wikitext2", "wikitext103"],
        help="Dataset to train and evaluate on.",
    )
    parser.add_argument(
        "--context_size",
        type=int,
        default=1024,
        help="Context size in tokens of the model.",
    )
    parser.add_argument(
        "--num_blocks", type=int, default=12, help="Number of Transformer blocks."
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=1024,
        help="Dimensionality of embeddings and hidden layers.",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=16,
        help="Number of heads used for multi-head attention",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Minibatch size.")
    parser.add_argument(
        "--num_iters", type=int, default=100000, help="Iterations to train for."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="SGD learning rate."
    )
    parser.add_argument(
        "--steps_per_report",
        type=int,
        default=10,
        help="Number of training steps between loss reporting.",
    )
    parser.add_argument(
        "--steps_per_eval",
        type=int,
        default=1000,
        help="Number of training steps between validations.",
    )
    parser.add_argument(
        "--eval_test",
        action="store_true",
        help="Evaluate on the test set after training",
    )
    args = parser.parse_args()
    if not args.gpu:
        mx.set_default_device(mx.cpu)
    main(args)
