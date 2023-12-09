# Copyright © 2023 Apple Inc.

import argparse
import json
import math
import numpy as np
from pathlib import Path
from sentencepiece import SentencePieceProcessor
import time
from typing import Optional, Tuple, List

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map, tree_flatten, tree_unflatten


from models import ModelArgs, Model, LoRALinear
import wikisql


def build_parser():
    parser = argparse.ArgumentParser(
        description="LoRA finetuning with Llama or Mistral"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="A path to the model files containing the tokenizer, weights, config.",
    )
    # Generation args
    parser.add_argument(
        "--num-tokens", "-n", type=int, default=100, help="How many tokens to generate"
    )
    parser.add_argument(
        "--write-every", type=int, default=1, help="After how many tokens to detokenize"
    )
    parser.add_argument(
        "--temp", type=float, default=0.8, help="The sampling temperature"
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        help="The prompt for generation",
        default=None,
    )

    # Training args
    parser.add_argument(
        "--train",
        action="store_true",
        help="Do training",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Minibatch size.")
    parser.add_argument(
        "--iters", type=int, default=1000, help="Iterations to train for."
    )
    parser.add_argument(
        "--val_batches",
        type=int,
        default=100,
        help="Number of validation batches, -1 uses the entire validation set.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Adam learning rate."
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
        default=200,
        help="Number of training steps between validations.",
    )
    parser.add_argument(
        "--resume_adapter_file",
        type=str,
        default=None,
        help="Load path to resume training with the given adapter weights.",
    )
    parser.add_argument(
        "--adapter_file",
        type=str,
        default="adapters.npz",
        help="Save/load path for the trained adapter weights.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate on the test set after training",
    )
    parser.add_argument(
        "--test_batches",
        type=int,
        default=500,
        help="Number of test set batches, -1 uses the entire test set.",
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")
    return parser


class Tokenizer:
    def __init__(self, model_path: str):
        assert Path(model_path).exists(), model_path
        self._model = SentencePieceProcessor(model_file=model_path)
        self._sep = "▁"
        assert self._model.vocab_size() == self._model.get_piece_size()

    def encode(self, s: str) -> List[int]:
        return [self._model.bos_id(), *self._model.encode(s)]

    def decode(self, t: List[int]) -> str:
        out = self._model.decode(t)
        if t and self._model.id_to_piece(t[0])[0] == self._sep:
            return " " + out
        return out

    @property
    def vocab_size(self) -> int:
        return self._model.vocab_size()


def loss(model, inputs, targets, lengths):
    # Run model on inputs
    logits, _ = model(inputs)

    # Mask padding tokens
    length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

    # Calculate the loss
    ce = nn.losses.cross_entropy(logits, targets) * length_mask
    ntoks = length_mask.sum()
    ce = ce.sum() / ntoks
    return ce, ntoks


def iterate_batches(dset, tokenizer, batch_size, shuffle=True):
    # Shuffle indices
    indices = np.arange(len(dset))
    if shuffle:
        indices = np.random.permutation(indices)

    # Collect batches from dataset
    for i in range(0, len(indices) - batch_size + 1, batch_size):
        # Encode batch
        batch = [tokenizer.encode(dset[indices[i + j]]) for j in range(batch_size)]
        lengths = [len(x) for x in batch]

        # Pad to the max length
        batch_arr = np.zeros((batch_size, max(lengths)), np.int32)
        for j in range(batch_size):
            batch_arr[j, : lengths[j]] = batch[j]
        batch = mx.array(batch_arr)
        yield batch[:, :-1], batch[:, 1:], mx.array(lengths)


def evaluate(model, dataset, loss, tokenizer, batch_size, num_batches):
    all_losses = []
    ntokens = 0
    for it, batch in zip(
        range(num_batches),
        iterate_batches(dataset, tokenizer, batch_size, shuffle=False),
    ):
        losses, toks = loss(model, *batch)
        all_losses.append((losses * toks).item())
        ntokens += toks.item()

    return np.sum(all_losses) / ntokens


def train(model, train_set, val_set, optimizer, loss, tokenizer, args):
    # Create value and grad function for loss
    loss_value_and_grad = nn.value_and_grad(model, loss)

    losses = []
    n_tokens = 0

    # Main training loop
    start = time.perf_counter()
    for it, batch in zip(
        range(args.iters), iterate_batches(train_set, tokenizer, args.batch_size)
    ):
        # Forward and backward pass
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)

        # Model update
        optimizer.update(model, grad)
        mx.eval(model.parameters(), optimizer.state, lvalue)

        # Record loss
        losses.append(lvalue.item())
        n_tokens += toks.item()

        # Report training loss if needed
        if (it + 1) % args.steps_per_report == 0:
            train_loss = np.mean(losses)

            stop = time.perf_counter()
            print(
                f"Iter {it + 1}: Train loss {train_loss:.3f}, "
                f"It/sec {args.steps_per_report / (stop - start):.3f}, "
                f"Tokens/sec {float(n_tokens) / (stop - start):.3f}"
            )
            losses = []
            n_tokens = 0
            start = time.perf_counter()

        # Report validation loss if needed
        if it == 0 or (it + 1) % args.steps_per_eval == 0:
            stop = time.perf_counter()
            val_loss = evaluate(
                model, val_set, loss, tokenizer, args.batch_size, args.val_batches
            )
            print(
                f"Iter {it + 1}: "
                f"Val loss {val_loss:.3f}, "
                f"Val took {(time.perf_counter() - stop):.3f}s"
            )

            start = time.perf_counter()


def generate(model, prompt, tokenizer, args):
    print(args.prompt, end="", flush=True)
    prompt = mx.array(tokenizer.encode(args.prompt))

    def generate_step():
        temp = args.temp

        def sample(logits):
            if temp == 0:
                return mx.argmax(logits, axis=-1)
            else:
                return mx.random.categorical(logits * (1 / temp))

        logits, cache = model(prompt[None])
        y = sample(logits[:, -1, :])
        yield y

        while True:
            logits, cache = model(y[:, None], cache)
            y = sample(logits.squeeze(1))
            yield y

    tokens = []
    for token, _ in zip(generate_step(), range(args.num_tokens)):
        tokens.append(token)

        if (len(tokens) % 10) == 0:
            mx.eval(tokens)
            s = tokenizer.decode([t.item() for t in tokens])
            print(s, end="", flush=True)
            tokens = []

    mx.eval(tokens)
    s = tokenizer.decode([t.item() for t in tokens])
    print(s, flush=True)


def load_model(folder: str, dtype=mx.float32):
    model_path = Path(folder)
    tokenizer = Tokenizer(str(model_path / "tokenizer.model"))
    with open(model_path / "params.json", "r") as f:
        config = json.loads(f.read())
        model_args = ModelArgs(**config)
        if config.get("vocab_size", -1) < 0:
            config["vocab_size"] = tokenizer.vocab_size
    weights = mx.load(str(model_path / "weights.npz"))
    weights = tree_unflatten(list(weights.items()))
    weights = tree_map(lambda p: p.astype(dtype), weights)
    model = Model(model_args)
    model.update(weights)
    return model, tokenizer


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("Loading pretrained model")
    model, tokenizer = load_model(args.model)

    # Freeze all layers other than LORA linears
    model.freeze()
    for l in model.layers[16:32]:
        l.attention.wq = LoRALinear.from_linear(l.attention.wq)
        l.attention.wv = LoRALinear.from_linear(l.attention.wv)

    p = sum(v.size for _, v in tree_flatten(model.parameters())) / 10**6
    print(f"Total parameters {p:.3f}M")
    p = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
    print(f"Trainable parameters {p:.3f}M")

    print("Loading datasets")
    train_set, valid_set, test_set = wikisql.load()

    # Resume training the given adapters.
    if args.resume_adapter_file is not None:
        print(f"Loading pretrained adapters from {args.resume_adapter_file}")
        model.load_weights(args.resume_adapter_file)

    if args.train:
        print("Training")
        opt = optim.Adam(learning_rate=args.learning_rate)

        # Train model
        train(model, train_set, valid_set, opt, loss, tokenizer, args)

        # Save adapter weights
        mx.savez(args.adapter_file, **dict(tree_flatten(model.trainable_parameters())))

    # Load the LoRA adapter weights which we assume should exist by this point
    model.load_weights(args.adapter_file)

    if args.test:
        print("Testing")

        test_loss = evaluate(
            model,
            test_set,
            loss,
            tokenizer,
            args.batch_size,
            num_batches=args.test_batches,
        )
        test_ppl = math.exp(test_loss)

        print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")

    if args.prompt is not None:
        print("Generating")
        generate(model, args.prompt, tokenizer, args)
