# Copyright © 2023 Apple Inc.

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import utils as lora_utils
from mlx.utils import tree_flatten, tree_unflatten
from models.lora import LoRALinear


def build_parser():
    parser = argparse.ArgumentParser(description="LoRA or QLoRA finetuning.")
    parser.add_argument(
        "--model",
        default="mlx_model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    # Generation args
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=100,
        help="The maximum number of tokens to generate",
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
    parser.add_argument(
        "--data",
        type=str,
        default="data/",
        help="Directory with {train, valid, test}.jsonl files",
    )
    parser.add_argument(
        "--lora-layers",
        type=int,
        default=16,
        help="Number of layers to fine-tune",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Minibatch size.")
    parser.add_argument(
        "--iters", type=int, default=1000, help="Iterations to train for."
    )
    parser.add_argument(
        "--val-batches",
        type=int,
        default=25,
        help="Number of validation batches, -1 uses the entire validation set.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-5, help="Adam learning rate."
    )
    parser.add_argument(
        "--steps-per-report",
        type=int,
        default=10,
        help="Number of training steps between loss reporting.",
    )
    parser.add_argument(
        "--steps-per-eval",
        type=int,
        default=200,
        help="Number of training steps between validations.",
    )
    parser.add_argument(
        "--resume-adapter-file",
        type=str,
        default=None,
        help="Load path to resume training with the given adapter weights.",
    )
    parser.add_argument(
        "--adapter-file",
        type=str,
        default="adapters.npz",
        help="Save/load path for the trained adapter weights.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save the model every N iterations.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate on the test set after training",
    )
    parser.add_argument(
        "--test-batches",
        type=int,
        default=500,
        help="Number of test set batches, -1 uses the entire test set.",
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")
    return parser


class Dataset:
    """
    Light-weight wrapper to hold lines from a jsonl file
    """

    def __init__(self, path: Path, key: str = "text"):
        if not path.exists():
            self._data = None
        else:
            with open(path, "r") as fid:
                self._data = [json.loads(l) for l in fid]
        self._key = key

    def __getitem__(self, idx: int):
        return self._data[idx][self._key]

    def __len__(self):
        return len(self._data)


def load(args):
    def load_and_check(name):
        dataset_path = Path(args.data) / f"{name}.jsonl"
        try:
            return Dataset(dataset_path)
        except Exception as e:
            print(f"Unable to build dataset {dataset_path} ({e})")
            raise

    names = ("train", "valid", "test")
    train, valid, test = (load_and_check(n) for n in names)

    if args.train and len(train) == 0:
        raise ValueError(
            "Training set not found or empty. Must provide training set for fine-tuning."
        )
    if args.train and len(valid) == 0:
        raise ValueError(
            "Validation set not found or empty. Must provide validation set for fine-tuning."
        )
    if args.test and len(test) == 0:
        raise ValueError(
            "Test set not found or empty. Must provide test set for evaluation."
        )
    return train, valid, test


def loss(model, inputs, targets, lengths):
    # Run model on inputs
    logits, _ = model(inputs)
    logits = logits.astype(mx.float32)

    # Mask padding tokens
    length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

    # Calculate the loss
    ce = nn.losses.cross_entropy(logits, targets) * length_mask
    ntoks = length_mask.sum()
    ce = ce.sum() / ntoks
    return ce, ntoks


def iterate_batches(dset, tokenizer, batch_size, train=False):
    # Shuffle indices
    while True:
        indices = np.arange(len(dset))
        if train:
            indices = np.random.permutation(indices)

        # Collect batches from dataset
        for i in range(0, len(indices) - batch_size + 1, batch_size):
            # Encode batch
            batch = [tokenizer.encode(dset[indices[i + j]]) for j in range(batch_size)]
            lengths = [len(x) for x in batch]

            # Check if any sequence is longer than 2048 tokens
            if max(lengths) > 2048:
                print(
                    "[WARNING] Some sequences are longer than 2048 tokens. "
                    "Consider pre-splitting your data to save memory."
                )

            # Pad to the max length
            batch_arr = np.zeros((batch_size, max(lengths)), np.int32)

            for j in range(batch_size):
                batch_arr[j, : lengths[j]] = batch[j]
            batch = mx.array(batch_arr)
            yield batch[:, :-1], batch[:, 1:], mx.array(lengths)

        if not train:
            break


def evaluate(model, dataset, loss, tokenizer, batch_size, num_batches):
    all_losses = []
    ntokens = 0
    for it, batch in zip(
        range(num_batches),
        iterate_batches(dataset, tokenizer, batch_size),
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
        range(args.iters),
        iterate_batches(train_set, tokenizer, args.batch_size, train=True),
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

        # Save adapter weights if needed
        if (it + 1) % args.save_every == 0:
            mx.savez(
                args.adapter_file, **dict(tree_flatten(model.trainable_parameters()))
            )
            print(f"Iter {it + 1}: Saved adapter weights to {args.adapter_file}.")


def generate(model, prompt, tokenizer, args):
    print(prompt, end="", flush=True)

    prompt = mx.array(tokenizer.encode(prompt))

    tokens = []
    skip = 0
    for token, n in zip(
        lora_utils.generate(prompt, model, args.temp),
        range(args.max_tokens),
    ):
        if token == tokenizer.eos_token_id:
            break

        tokens.append(token.item())
        s = tokenizer.decode(tokens)
        if len(s) - skip > 1:
            print(s[skip:-1], end="", flush=True)
            skip = len(s) - 1
    print(tokenizer.decode(tokens)[skip:], flush=True)
    print("=" * 10)
    if len(tokens) == 0:
        print("No tokens generated for this prompt")
        return


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("Loading pretrained model")
    model, tokenizer, _ = lora_utils.load(args.model)

    # Freeze all layers other than LORA linears
    model.freeze()
    for l in model.model.layers[len(model.model.layers) - args.lora_layers :]:
        l.self_attn.q_proj = LoRALinear.from_linear(l.self_attn.q_proj)
        l.self_attn.v_proj = LoRALinear.from_linear(l.self_attn.v_proj)
        if hasattr(l, "block_sparse_moe"):
            l.block_sparse_moe.gate = LoRALinear.from_linear(l.block_sparse_moe.gate)

    p = sum(v.size for _, v in tree_flatten(model.parameters())) / 10**6
    print(f"Total parameters {p:.3f}M")
    p = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
    print(f"Trainable parameters {p:.3f}M")

    print("Loading datasets")
    train_set, valid_set, test_set = load(args)

    # Resume training the given adapters.
    if args.resume_adapter_file is not None:
        print(f"Loading pretrained adapters from {args.resume_adapter_file}")
        model.load_weights(args.resume_adapter_file, strict=False)

    if args.train:
        print("Training")
        opt = optim.Adam(learning_rate=args.learning_rate)

        # Train model
        train(model, train_set, valid_set, opt, loss, tokenizer, args)

        # Save adapter weights
        mx.savez(args.adapter_file, **dict(tree_flatten(model.trainable_parameters())))

    # Load the LoRA adapter weights which we assume should exist by this point
    if not Path(args.adapter_file).is_file():
        raise ValueError(
            f"Adapter file {args.adapter_file} missing. "
            "Use --train to learn and save the adapters.npz."
        )
    model.load_weights(args.adapter_file, strict=False)

    if args.test:
        print("Testing")
        model.eval()
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
