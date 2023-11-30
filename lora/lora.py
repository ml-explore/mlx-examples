# Copyright © 2023 Apple Inc.

import argparse
import math
import numpy as np
from sentencepiece import SentencePieceProcessor
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten


from llama import LoRALinear, load_model
import wikisql


def build_parser():
    parser = argparse.ArgumentParser(description="Llama LoRA finetuning")
    parser.add_argument(
        "--model", required=True, help="The model file containing MLX weights"
    )
    parser.add_argument(
        "--tokenizer", required=True, help="The sentencepiece tokenizer"
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


def loss(model, inputs, targets, lengths):
    # Run model on inputs
    logits = model(inputs)

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
        batch = tokenizer.encode([dset[indices[i + j]] for j in range(batch_size)])
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
    # Encode prompt
    x = mx.array([[tokenizer.bos_id()] + tokenizer.encode(prompt)])

    skip = 0
    prompt_processing = None
    tokens = []

    # Genertation loop
    start = time.perf_counter()
    for token in model.generate(x, args.temp):
        tokens.append(token)

        if len(tokens) == 1:
            # Actually perform the computation to measure the prompt processing time
            mx.eval(token)
            prompt_processing = time.perf_counter() - start

        if len(tokens) >= args.num_tokens:
            break

        if (len(tokens) % args.write_every) == 0:
            mx.eval(tokens)
            s = tokenizer.decode([t.item() for t in tokens])
            print(s[skip:], end="", flush=True)
            skip = len(s)

    mx.eval(tokens)
    full_gen = time.perf_counter() - start

    s = tokenizer.decode([t.item() for t in tokens])
    print(s[skip:], end="", flush=True)
    print()
    print(f"Prompt processing took: {prompt_processing:.3f} s")
    print(f"Full generation took: {full_gen:.3f} s")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("Loading tokenizer")
    tokenizer = SentencePieceProcessor(model_file=args.tokenizer)

    print("Loading pretrained model")
    model = load_model(args.model)

    # Freeze all layers other than LORA linears
    model.freeze()
    for l in model.layers[16:32]:
        l.attention.query_proj = LoRALinear.from_linear(l.attention.query_proj)
        l.attention.value_proj = LoRALinear.from_linear(l.attention.value_proj)

    p = sum(v.size for _, v in tree_flatten(model.parameters())) / 10**6
    print(f"Total parameters {p:.3f}M")
    p = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
    print(f"Trainable parameters {p:.3f}M")

    print("Loading datasets")
    train_set, valid_set, test_set = wikisql.load()

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
