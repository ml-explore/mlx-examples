# Copyright © 2024 Apple Inc.

import glob
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten

from trainer import TrainingArgs, TrainingCallback, grad_checkpoint



def compute_ppo_loss(
    new_logprobs: mx.array,
    old_logprobs: mx.array, 
    values: mx.array,
    old_values: mx.array,
    advantages: mx.array,
    returns: mx.array,
    padding_mask: mx.array,
    padding_mask_p1: mx.array = None,
    vf_coef: float = 0.5,
    cliprange: float = 0.2,
    cliprange_value: float = 0.2
) -> tuple[mx.array, mx.array, mx.array]:
    """Compute PPO loss with policy and value components and masking"""
    padding_mask_p1 = padding_mask_p1 if padding_mask_p1 is not None else padding_mask
    
    # Value loss 
    vpred_clipped = mx.clip(values, old_values - cliprange_value, old_values + cliprange_value)
    vf_losses = mx.maximum(
        mx.square(values - returns),
        mx.square(vpred_clipped - returns)
    )
    vf_loss = 0.5 * mx.mean(mx.where(~padding_mask_p1, vf_losses, 0))
    
    # Policy loss
    ratio = mx.exp(new_logprobs - old_logprobs)
    pg_losses = mx.maximum(
        -advantages * ratio,
        -advantages * mx.clip(ratio, 1.0 - cliprange, 1.0 + cliprange)
    )
    pg_loss = mx.mean(mx.where(~padding_mask, pg_losses, 0))
    
    total_loss = pg_loss + vf_coef * vf_loss
    return total_loss, pg_loss, vf_loss


@dataclass
class PPOTrainingArgs(TrainingArgs):
    vf_coef: float = field(default=0.5, metadata={"help": "Value function coefficient"})
    cliprange: float = field(default=0.2, metadata={"help": "Policy gradient clipping range"}) 
    cliprange_value: float = field(default=0.2, metadata={"help": "Value function clipping range"})


def ppo_loss(
    model,
    inputs,
    targets,
    lengths,
    old_logprobs,
    values,
    old_values,
    advantages,
    returns,
    vf_coef=0.5,
    cliprange=0.2,
    cliprange_value=0.2
):
   # Get new logits and create length mask
   logits = model(inputs).astype(mx.float32)
   length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]
   
   # Get new log probs
   new_logprobs = nn.losses.cross_entropy(logits, targets) * length_mask
   ntoks = length_mask.sum()
   new_logprobs = new_logprobs.sum() / ntoks

   # Value loss with clipping
   vpred_clipped = mx.clip(values, old_values - cliprange_value, old_values + cliprange_value)
   vf_loss = 0.5 * mx.maximum(
       mx.square(values - returns),
       mx.square(vpred_clipped - returns)
   ).mean() 

   # Policy loss with clipping
   ratio = mx.exp(new_logprobs - old_logprobs)
   pg_loss = mx.maximum(
       -advantages * ratio,
       -advantages * mx.clip(ratio, 1.0 - cliprange, 1.0 + cliprange)
   ).mean()

   total_loss = pg_loss + vf_coef * vf_loss
   return total_loss, pg_loss, vf_loss, ntoks


def iterate_batches(dataset, tokenizer, batch_size, max_seq_length, train=False):
    # Sort by length:
    idx = sorted(range(len(dataset)), key=lambda idx: len(dataset[idx]))
    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size}"
            f" examples but only has {len(dataset)}."
        )

    # If running in distributed mode (N machines) then each one should skip N-1
    # samples
    step = mx.distributed.init().size()
    if batch_size % step != 0:
        raise ValueError("The batch size must be divisible by the number of workers")

    # Make the batches:
    batch_idx = [
        idx[i : i + batch_size : step]
        for i in range(0, len(idx) - batch_size + 1, batch_size)
    ]

    while True:
        indices = np.random.permutation(len(batch_idx))
        for i in indices:
            batch = [dataset[j] for j in batch_idx[i]]
            lengths = [len(x) for x in batch]
            if max(lengths) > max_seq_length:
                print(
                    f"[WARNING] Some sequences are longer than {max_seq_length} tokens. "
                    f"The longest sentence {max(lengths)} will be truncated to {max_seq_length}. "
                    "Consider pre-splitting your data to save memory."
                )

            # Pad to the nearest multiple of 8 or the maximum length
            pad_to = 8
            max_length_in_batch = pad_to * ((max(lengths) + pad_to - 1) // pad_to)
            max_length_in_batch = min(max_length_in_batch, max_seq_length)

            batch_arr = np.zeros((batch_size // step, max_length_in_batch), np.int32)

            for j in range(batch_size // step):
                truncated_length = min(lengths[j], max_seq_length)
                batch_arr[j, :truncated_length] = batch[j][:truncated_length]
                lengths[j] = (
                    truncated_length  # Update lengths to match truncated lengths
                )
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
   old_logprobs=None,  
   values=None,
   old_values=None,
   advantages=None, 
   returns=None,
   vf_coef=0.5,
   cliprange=0.2, 
   cliprange_value=0.2,
   loss: callable = compute_ppo_loss,
   iterate_batches: callable = iterate_batches,
):
   total_loss = 0
   total_pg_loss = 0 
   total_vf_loss = 0
   ntokens = 0

   index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)

   for _, batch in zip(
       index_iterator,
       iterate_batches(
           dataset=dataset,
           tokenizer=tokenizer, 
           batch_size=batch_size,
           max_seq_length=max_seq_length,
       ),
   ):
       losses, pg_loss, vf_loss, toks = loss(
           model, *batch,
           old_logprobs=old_logprobs,
           values=values,
           old_values=old_values, 
           advantages=advantages,
           returns=returns,
           vf_coef=vf_coef,
           cliprange=cliprange,
           cliprange_value=cliprange_value
       )
       
       total_loss += losses * toks
       total_pg_loss += pg_loss * toks
       total_vf_loss += vf_loss * toks
       ntokens += toks
       mx.eval(total_loss, total_pg_loss, total_vf_loss, ntokens)

   total_loss = mx.distributed.all_sum(total_loss, stream=mx.cpu)
   total_pg_loss = mx.distributed.all_sum(total_pg_loss, stream=mx.cpu)
   total_vf_loss = mx.distributed.all_sum(total_vf_loss, stream=mx.cpu)
   ntokens = mx.distributed.all_sum(ntokens, stream=mx.cpu)

   return (total_loss / ntokens).item(), (total_pg_loss / ntokens).item(), (total_vf_loss / ntokens).item()


def train(
    model,
    tokenizer,
    optimizer,
    train_dataset,
    val_dataset,
    args: TrainingArgs = TrainingArgs(),
    loss: callable = ppo_loss,
    iterate_batches: callable = iterate_batches,
    training_callback: TrainingCallback = None,
):
    print(f"Starting training..., iters: {args.iters}")
    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()
    if world_size > 1:
        print(f"Node {rank} of {world_size}")

    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    state = [model.state, optimizer.state]

    def step(batch):
        # Forward and backward pass
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)

        # All reduce the gradients if running in distributed mode
        grad = average_gradients(grad)

        # Model update
        optimizer.update(model, grad)

        return lvalue, toks

    loss_value_and_grad = nn.value_and_grad(model, loss)

    losses = 0
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    # Main training loop
    start = time.perf_counter()
    for it, batch in zip(
        range(1, args.iters + 1),
        iterate_batches(
            dataset=train_dataset,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            train=True,
        ),
    ):
        # Report validation loss if needed, the first validation loss
        # is always measured before any training.
        if it == 1 or it % args.steps_per_eval == 0 or it == args.iters:
            stop = time.perf_counter()
            val_loss = evaluate(
                model=model,
                dataset=val_dataset,
                loss=loss,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                iterate_batches=iterate_batches,
            )
            val_time = time.perf_counter() - stop
            if rank == 0:
                print(
                    f"Iter {it}: "
                    f"Val loss {val_loss:.3f}, "
                    f"Val took {val_time:.3f}s",
                    flush=True,
                )

            if training_callback is not None:
                val_info = {
                    "iteration": it,
                    "val_loss": val_loss,
                    "val_time": val_time,
                }
                training_callback.on_val_loss_report(val_info)

            start = time.perf_counter()

        lvalue, toks = step(batch)
        losses += lvalue
        n_tokens += toks
        steps += 1
        mx.eval(state, losses, n_tokens)

        # Report training loss if needed
        if it % args.steps_per_report == 0 or it == args.iters:
            stop = time.perf_counter()

            train_loss = mx.distributed.all_sum(losses, stream=mx.cpu).item()
            train_loss /= steps * mx.distributed.init().size()
            n_tokens = mx.distributed.all_sum(n_tokens, stream=mx.cpu).item()
            learning_rate = optimizer.learning_rate.item()
            it_sec = args.steps_per_report / (stop - start)
            tokens_sec = float(n_tokens) / (stop - start)
            trained_tokens += n_tokens
            peak_mem = mx.metal.get_peak_memory() / 1e9
            if rank == 0:
                print(
                    f"Iter {it}: Train loss {train_loss:.3f}, "
                    f"Learning Rate {learning_rate:.3e}, "
                    f"It/sec {it_sec:.3f}, "
                    f"Tokens/sec {tokens_sec:.3f}, "
                    f"Trained Tokens {trained_tokens}, "
                    f"Peak mem {peak_mem:.3f} GB",
                    flush=True,
                )

            if training_callback is not None:
                train_info = {
                    "iteration": it,
                    "train_loss": train_loss,
                    "learning_rate": learning_rate,
                    "iterations_per_second": it_sec,
                    "tokens_per_second": tokens_sec,
                    "trained_tokens": trained_tokens,
                    "peak_memory": peak_mem,
                }
                training_callback.on_train_loss_report(train_info)

            losses = 0
            n_tokens = 0
            steps = 0
            start = time.perf_counter()

        # Save adapter weights
        if it % args.steps_per_save == 0:
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(args.adapter_file), adapter_weights)
            checkpoint = (
                Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
            )
            mx.save_safetensors(str(checkpoint), adapter_weights)
            print(
                f"Iter {it}: Saved adapter weights to "
                f"{args.adapter_file} and {checkpoint}."
            )

    # Save final weights
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(args.adapter_file), adapter_weights)
    print(f"Saved final weights to {args.adapter_file}.")
