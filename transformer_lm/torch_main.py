# Copyright © 2023 Apple Inc.

import math
import time

import datasets
import numpy as np
import torch


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


def create_additive_causal_mask(N, device):
    # torch.nn.Transformer.generate_square_subsequent_mask
    # gives NaNs with `device="mps"`
    indices = torch.arange(N, device=device)
    mask = indices.reshape((-1, 1)) < indices.reshape((1, -1))
    return mask.to(torch.float32) * -1e9


class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size, num_layers, num_heads, model_dims):
        super().__init__()

        self.embedding = torch.nn.Embedding(vocab_size, model_dims)
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                model_dims,
                num_heads,
                4 * model_dims,
                dropout=0.0,
                norm_first=True,
                batch_first=True,
            ),
            num_layers,
        )
        self.projection = torch.nn.Linear(model_dims, vocab_size)

    def forward(self, x):
        mask = create_additive_causal_mask(x.shape[1], device=x.device)
        x = self.embedding(x)
        x = self.transformer(x, mask=mask)
        x = self.projection(x)
        return x


def main(args, device):
    batch_size = args.batch_size
    context_size = args.context_size
    steps_per_eval = args.steps_per_eval
    steps_per_report = args.steps_per_report

    # Load vocab and dataset:
    vocab, train, valid, test = datasets.ptb()

    # Initialize model:
    transformer = TransformerLM(len(vocab), args.num_blocks, args.num_heads, args.dim)
    transformer = transformer.to(device)
    optim = torch.optim.SGD(transformer.parameters(), lr=args.learning_rate, momentum=0)
    nparams = sum(
        p.numel() for n, p in transformer.named_parameters() if "embedding" not in n
    )
    print(f"Training a transformer with {nparams / 1024**2:.3f} M parameters")

    @torch.no_grad()
    def eval_fn(dataset):
        inputs, targets = to_samples(context_size, dataset)
        loss = 0
        for s in range(0, targets.shape[0], batch_size):
            bx, by = inputs[s : s + batch_size], targets[s : s + batch_size]
            bx, by = map(lambda x: torch.from_numpy(x.astype(int)).to(device), [bx, by])
            logits = transformer(bx)
            losses = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), by.flatten(), reduction="none"
            )
            losses = losses.view(-1, by.shape[-1]).mean(-1)
            loss += losses.sum().item()
        return loss / len(targets)

    train_iterator = iterate_batches(batch_size, context_size, train)
    losses = []
    tic = time.perf_counter()
    for it, (inputs, targets) in zip(range(args.num_iters), train_iterator):
        inputs, targets = map(
            lambda x: torch.from_numpy(x.astype(int)).to(device), [inputs, targets]
        )
        optim.zero_grad()
        logits = transformer(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), targets.flatten()
        )
        loss.backward()
        optim.step()
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
            val_loss = eval_fn(valid)
            toc = time.perf_counter()
            print(
                f"Iter {it + 1}: "
                f"Val loss {val_loss:.3f}, "
                f"Val ppl {math.exp(val_loss):.3f}, "
                f"Val took {(toc - tic):.3f}s, "
            )
            tic = time.perf_counter()

    if args.eval_test:
        test_loss = eval_fn(test)
        test_ppl = math.exp(test_loss)
        print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Train a decoder-only Transformer LM with MLX.")
    parser.add_argument("--gpu", action="store_true", help="Use the Metal back-end.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the RNGs.")
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
    main(args, device="mps" if args.gpu else "cpu")
