import os
import time
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten


@dataclass
class TrainingArgs:
    lora_layers: int = field(
        default=16, metadata={"help": "Number of layers to fine-tune"}
    )
    batch_size: int = field(default=4, metadata={"help": "Minibatch size."})
    iters: int = field(default=100, metadata={"help": "Iterations to train for."})
    val_batches: int = field(
        default=25,
        metadata={
            "help": "Number of validation batches, -1 uses the entire validation set."
        },
    )
    steps_per_report: int = field(
        default=10,
        metadata={"help": "Number of training steps between loss reporting."},
    )
    steps_per_eval: int = field(
        default=200, metadata={"help": "Number of training steps between validations."}
    )
    steps_per_save: int = field(
        default=100, metadata={"help": "Save the model every number steps"}
    )
    max_seq_length: int = field(
        default=2048, metadata={"help": "Maximum sequence length."}
    )
    adapter_file: str = field(
        default="adapter.npz",
        metadata={"help": "Save/load path for the trained adapter weights."},
    )


def default_loss(model, inputs, targets, lengths):
    logits, _ = model(inputs)
    logits = logits.astype(mx.float32)

    length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

    ce = nn.losses.cross_entropy(logits, targets) * length_mask
    ntoks = length_mask.sum()
    ce = ce.sum() / ntoks

    return ce, ntoks


def iterate_batches(dataset, tokenizer, batch_size, max_seq_length, train=False):
    while True:
        # Shuffle indices
        indices = np.arange(len(dataset))
        indices = np.random.permutation(indices)
        # Collect batches from dataset
        for i in range(0, len(indices) - batch_size + 1, batch_size):
            # Encode batch
            batch = [
                tokenizer.encode(dataset[indices[i + j]]) for j in range(batch_size)
            ]
            lengths = [len(x) for x in batch]

            if max(lengths) > max_seq_length:
                print(
                    f"[WARNING] Some sequences are longer than {max_seq_length} tokens. "
                    f"The longest sentence {max(lengths)} will be truncated to {max_seq_length}. "
                    "Consider pre-splitting your data to save memory."
                )

            # Pad to the max length
            max_length_in_batch = min(max(lengths), max_seq_length)
            batch_arr = np.zeros((batch_size, max_length_in_batch), np.int32)

            for j in range(batch_size):
                truncated_length = min(lengths[j], max_seq_length)
                batch_arr[j, :truncated_length] = batch[j][:truncated_length]
                lengths[
                    j
                ] = truncated_length  # Update lengths to match truncated lengths
            batch = mx.array(batch_arr)

            yield batch[:, :-1], batch[:, 1:], mx.array(lengths)

        if not train:
            break


def evaluate(
    model,
    dataset,
    tokenizer,
    batch_size,
    num_batches,
    max_seq_length=2048,
    loss: callable = default_loss,
):
    all_losses = []
    ntokens = 0
    for it, batch in zip(
        range(num_batches),
        iterate_batches(
            dataset=dataset,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        ),
    ):
        losses, toks = loss(model, *batch)
        all_losses.append((losses * toks).item())
        ntokens += toks.item()

    return np.sum(all_losses) / ntokens


def train(
    model,
    tokenizer,
    optimizer,
    train_dataset,
    val_dataset,
    args: TrainingArgs = TrainingArgs(),
    loss: callable = default_loss,
):
    # Create value and grad function for loss
    loss_value_and_grad = nn.value_and_grad(model, loss)

    losses = []
    n_tokens = 0
    print("Starting training..., iters:", args.iters)
    # Main training loop
    start = time.perf_counter()
    for it, batch in zip(
        range(args.iters),
        iterate_batches(
            dataset=train_dataset,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            train=True,
        ),
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
                model=model,
                dataset=val_dataset,
                loss=loss,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
            )
            print(
                f"Iter {it + 1}: "
                f"Val loss {val_loss:.3f}, "
                f"Val took {(time.perf_counter() - stop):.3f}s"
            )

            start = time.perf_counter()

            # Save adapter weights if needed
            if (it + 1) % args.steps_per_save == 0:
                save_adapter(model=model, adapter_file=args.adapter_file)
                print(
                    f"Iter {it + 1}: Saved adapter weights to {os.path.join(args.adapter_file)}."
                )
    # save final adapter weights
    save_adapter(model=model, adapter_file=args.adapter_file)
    print(f"Saved final adapter weights to {os.path.join(args.adapter_file)}.")


def save_adapter(
    model: nn.Module,
    adapter_file: str,
):
    flattened_tree = tree_flatten(model.trainable_parameters())

    mx.savez(adapter_file, **dict(flattened_tree))
