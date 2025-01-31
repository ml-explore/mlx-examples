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
from mlx.utils import tree_flatten

from .trainer import grad_checkpoint, TrainingArgs, TrainingCallback, average_gradients

from mlx_lm import generate

generate()

@dataclass
class GRPOTrainingArgs(TrainingArgs):
    group_size: int = field(
        default=4,
        metadata={"help": "Number of response sper prompt."},
    )
    is_reference_free: bool = field(
        default=False,
        metadata={
            "help": "Whether to use reference-free DPO training."
        }
    )
    beta: float = field(
        default=0.1, metadata={"help": "KL penalty coefficient."}
    )
    epsilon: float = field(
        default=1e-4, metadata={"help": "The Epsilon for numerical stability."}
    )
    reference_model_path: str = field(
        default=None,
        metadata={
            "help": "Path to reference model weights. If None, uses the same model."
        }
    )


def compute_default_rewards(sequences, batch_size, group_size):
    """
    Args:
        sequences: List of word sequences
        batch_size: Number of original prompts
        group_size: Number of generations per prompt
    """
    rewards = mx.zeros((len(sequences),))
    
    for i, sequence in enumerate(sequences):
        # Convert sequence to list if it isn't already
        if not isinstance(sequence, list):
            sequence = sequence.split()
            
        # Get the target (reversed) sequence
        target = sequence[::-1]
        
        # Calculate accuracy of reversal
        correct_positions = sum(1 for a, b in zip(sequence, target) if a == b)
        rewards[i] = correct_positions / len(sequence)
    
    return rewards


def grpo_loss(
        model,
        tokenizer,
        prompts,
        reward_funcs=None,
        beta=0.1,
        group_size=4,
        epslion=1e-4,
        ref_model = None
    ):
    batch_size = len(prompts)
    # Generate multiple completions for each prompt
    all_completions = []
    
    for prompt in prompts:
        prompt_completions = []
        for _ in range(group_size):
            completion = generate(model, tokenizer, prompt)
            prompt_completions.append(completion)
        all_completions.extend(prompt_completions)
    
    # Tokenize all prompts + completions (needed for model processing)
    tokenized_inputs = tokenizer(
        [p + c for p, c in zip(prompts * group_size, all_completions)],
        return_tensors="np",
        padding=True
    )
    
    inputs = mx.array(tokenized_inputs["input_ids"])
    attention_mask = mx.array(tokenized_inputs["attention_mask"])
    
    # Get lengths for proper masking
    lengths = attention_mask.sum(axis=1)
    
    # Get logits from current model
    logits = model(inputs).astype(mx.float32)
    
    # Calculate log probabilities
    log_probs = mx.log_softmax(logits[:, :-1, :], axis=-1)
    
    # Prepare targets (shift input_ids left by one position)
    targets = inputs[:, 1:]
    
    # Gather actual token probabilities
    token_log_probs = mx.take_along_axis(
        log_probs,
        targets.reshape(*targets.shape, 1),
        axis=-1
    ).squeeze(-1)
    
    # Get reference model log probabilities
    if ref_model is not None:
        ref_logits = ref_model(inputs).astype(mx.float32)
    else:
        ref_logits = model(inputs).astype(mx.float32)
        
    ref_log_probs = mx.log_softmax(ref_logits[:, :-1, :], axis=-1)
    ref_token_log_probs = mx.take_along_axis(
        ref_log_probs,
        targets.reshape(*targets.shape, 1),
        axis=-1
    ).squeeze(-1)
    
    # Compute the KL divergence between the model and the reference model
    kl_div = (mx.exp(ref_token_log_probs - token_log_probs) - (ref_token_log_probs - token_log_probs) - 1)

    # Calculate rewards
    if reward_funcs:
        rewards = mx.array([sum(rf(all_completions) for rf in reward_funcs)])
    else:
        rewards = compute_default_rewards(all_completions, batch_size, group_size)

    # Compute grouped-wise rewards
    grouped_rewards = rewards.reshape(batch_size, group_size)
    mean_grouped_rewards = mx.mean(grouped_rewards, axis=1)
    std_grouped_rewards = mx.std(grouped_rewards, axis=1)

    # Normalize the rewards to compute the advantages
    mean_grouped_rewards = mx.repeat(mean_grouped_rewards.reshape(-1, 1), group_size, axis=1).reshape(-1)
    std_grouped_rewards = mx.repeat(std_grouped_rewards.reshape(-1, 1), group_size, axis=1).reshape(-1)
    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + epslion)

    # Create length mask for the shifted sequence
    length_mask = mx.arange(inputs.shape[1] - 1)[None, :] < (lengths[:, None] - 1)

    # Calculate policy gradient loss, mx.stop_gradient allows for preserving gradients from token_log_probs
    per_token_loss = mx.exp(token_log_probs - mx.stop_gradient(token_log_probs)) * advantages.reshape(-1, 1)
    per_token_loss = -(per_token_loss - beta * kl_div)

    # Normalize loss properly per sequence
    sequence_sums = (per_token_loss * length_mask).sum(axis=1)
    sequence_lengths = length_mask.sum(axis=1)
    loss = (sequence_sums / sequence_lengths).mean()

    # Calculate mean KL divergence (normalized per sequence)
    mean_kl = ((kl_div * length_mask).sum(axis=1) / length_mask.sum(axis=1)).mean()
    metrics = {
        'rewards': rewards,
        'rewards_std': mx.std(rewards),
        'grouped_rewards': grouped_rewards,
        'grouped_rewards_std': mx.std(grouped_rewards),
        'kl': mean_kl
    }

    return loss, sequence_lengths.sum(), metrics


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


def evaluate_grpo(
    model,
    ref_model,
    dataset,
    tokenizer,
    batch_size,
    num_batches,
    beta: float,
    epslion: float,
    group_size: int,
    max_seq_length,
    reward_funcs = None,
    loss: callable = grpo_loss,
    iterate_batches: callable = iterate_batches
):
    all_losses = 0
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
        prompts = batch
        losses, toks, metrics = loss(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            reward_funcs=reward_funcs,
            beta=beta,
            group_size=group_size,
            epslion=epslion,
            ref_model=ref_model
        )
        all_losses += losses * toks
        ntokens += toks

        if all_metrics is None:
            all_metrics = {k: v * toks for k, v in metrics.items()}
        else:
            for k, v in metrics.items():
                all_metrics[k] += v * toks

        mx.eval(all_losses, ntokens)

    all_losses = mx.distributed.all_sum(all_losses, stream=mx.cpu)
    ntokens = mx.distributed.all_sum(ntokens, stream=mx.cpu)
    all_metrics = {k: mx.distributed.all_sum(v) for k, v in all_metrics.items()}

    avg_metrics = {k: (v / ntokens).item() for k, v in all_metrics.items()}
    avg_loss = (all_losses / ntokens).item()

    return avg_loss, ntokens, avg_metrics


def train_grpo(
    model,
    tokenizer,
    optimizer,
    train_dataset,
    val_dataset,
    args: GRPOTrainingArgs = GRPOTrainingArgs(),
    loss: callable = grpo_loss,
    iterate_batches: callable = iterate_batches,
    training_callback: TrainingCallback = None,
):
    print(f"Starting GRPO training..., iters: {args.iters}")
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
        (loss, toks, metrics), grad = loss_value_and_grad(model, *batch)

        # All reduce the gradients if running in distributed mode
        grad = average_gradients(grad)

        # Model update
        optimizer.update(model, grad)

        return loss, toks, metrics

    loss_value_and_grad = nn.value_and_grad(model, loss)

    losses = 0
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    accumulated_metrics = {
        'rewards': 0,
        'rewards_std': 0,
        'grouped_rewards': 0,
        'grouped_rewards_std': 0,
        'kl': 0
    }

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
            val_loss, val_ntokens, val_metrics = evaluate_grpo(
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
                    f"Val loss {val_loss:.8f}, "
                    f"Val rewards {val_metrics['rewards']:.3f}, "
                    f"Val rewards_std {val_metrics['rewards_std']:.3f}, "
                    f"Val grouped_rewards {val_metrics['grouped_rewards']:.3f}, "
                    f"Val grouped_rewards_std {val_metrics['grouped_rewards_std']:.3f}, "
                    f"Val kl {val_metrics['kl']:.3f}, "
                    f"Val took {val_time:.3f}s",
                    flush=True,
                )

            if training_callback is not None:
                training_callback.on_val_loss_report({
                    "iteration": it,
                    "val_loss": val_loss,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    "val_time": val_time,
                })

            start = time.perf_counter()

        loss, toks, metrics = step(batch)
        losses += loss
        n_tokens += toks
        steps += 1
        for k, v in metrics.items():
            accumulated_metrics[k] += v
        mx.eval(state, losses, n_tokens)

        if it % args.steps_per_report == 0 or it == args.iters:
            stop = time.perf_counter()

            train_loss = mx.distributed.all_sum(losses, stream=mx.cpu).item()
            train_loss /= steps * mx.distributed.init().size()
            avg_metrics = {k: v / (steps * world_size) for k, v in accumulated_metrics.items()}
            n_tokens = mx.distributed.all_sum(n_tokens, stream=mx.cpu).item()
            learning_rate = optimizer.learning_rate.item()
            it_sec = args.steps_per_report / (stop - start)
            tokens_sec = float(n_tokens) / (stop - start)
            trained_tokens += n_tokens
            peak_mem = mx.metal.get_peak_memory() / 1e9

            if rank == 0:
                print(
                    f"Iter {it}: Train loss {train_loss:.8f}, "
                    f"Rewards {avg_metrics['rewards']:.3f}, "
                    f"Rewards_std {avg_metrics['rewards_std']:.3f}, "
                    f"Grouped Rewards {avg_metrics['grouped_rewards']:.3f}, "
                    f"Grouped Rewards {avg_metrics['grouped_rewards']:.3f}, "
                    f"Grouped Rewards_std {val_metrics['grouped_rewards_std']:.3f}, "
                    f"KL {val_metrics['kl']:.3f}, "
                    f"Learning Rate {learning_rate:.3e}, "
                    f"It/sec {it_sec:.3f}, "
                    f"Tokens/sec {tokens_sec:.3f}, "
                    f"Peak mem {peak_mem:.3f} GB",
                    flush=True,
                )

            if training_callback is not None:
                training_callback.on_train_loss_report({
                    "iteration": it,
                    "train_loss": train_loss,
                    **{f"train_{k}": v for k, v in avg_metrics.items()},
                    "learning_rate": learning_rate,
                    "iterations_per_second": it_sec,
                    "tokens_per_second": tokens_sec,
                    "trained_tokens": trained_tokens,
                    "peak_memory": peak_mem,
                })

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