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
from ..generate import generate
from .trainer import TrainingCallback, grad_checkpoint, TrainingArgs


@dataclass
class DPOTrainingArgs(TrainingArgs):
    beta: float = field(
        default=0.1,
        metadata={"help": "Temperature parameter for DPO training."}
    )
    loss_type: str = field(
        default="sigmoid",
        metadata={
            "help": "DPO loss type: 'sigmoid', 'hinge', 'ipo', or 'dpop'."
        }
    )
    is_reference_free: bool = field(
        default=False,
        metadata={
            "help": "Whether to use reference-free DPO training."
        }
    )
    delta: float = field(
        default=50.0,
        metadata={
            "help": "Delta parameter for DPOP loss type."
        }
    )
    reference_model_path: str = field(
        default=None,
        metadata={
            "help": "Path to reference model weights. If None, uses the same model."
        }
    )


def dpo_loss(
    model,
    reference_teacher_model,
    chosen: mx.array,
    rejected: mx.array,
    chosen_masks: mx.array,
    rejected_masks: mx.array,
    beta: float,
    delta: float,
    loss_type: str = "sigmoid",
    is_reference_free: bool = False
):
    def make_predictions(model, x, mask):
        inputs = x[:, :-1]
        targets = x[:, 1:]
        
        logits = model(inputs)
        logits = logits.astype(mx.float32)

        return -nn.losses.cross_entropy(logits, targets) * mask[:, :-1]

    num_chosen_tokens = chosen_masks.sum(-1)
    num_rejected_tokens = rejected_masks.sum(-1)

    # Calculate log probabilities for policy model
    policy_chosen_scores = make_predictions(model, chosen, chosen_masks)
    policy_rejected_scores = make_predictions(model, rejected, rejected_masks)
    if loss_type == "ipo":
        # ipo uses average log probabilities
        policy_chosen_score = policy_chosen_scores.sum(-1) / num_chosen_tokens
        policy_rejected_score = policy_rejected_scores.sum(-1) / num_rejected_tokens
    else:
        policy_chosen_score = policy_chosen_scores.sum(-1)
        policy_rejected_score = policy_rejected_scores.sum(-1)

    # Calculate log probabilities for reference model
    if is_reference_free:
        reference_chosen_score = mx.zeros_like(policy_chosen_score)
        reference_rejected_score = mx.zeros_like(policy_rejected_score)
    else:
        reference_chosen_scores = mx.stop_gradient(make_predictions(reference_teacher_model, chosen, chosen_masks))
        reference_rejected_scores = mx.stop_gradient(make_predictions(reference_teacher_model, rejected, rejected_masks))
        if loss_type == "ipo":
            # ipo uses average log probabilities
            reference_chosen_score = reference_chosen_scores.sum(-1) / num_chosen_tokens
            reference_rejected_score = reference_rejected_scores.sum(-1) / num_rejected_tokens
        else:
            reference_chosen_score = reference_chosen_scores.sum(-1)
            reference_rejected_score = reference_rejected_scores.sum(-1)
    
    logits = (policy_chosen_score - policy_rejected_score) - (reference_chosen_score - reference_rejected_score)

    if loss_type == "sigmoid": # From the og paper
        losses = -nn.log_sigmoid(beta * logits)
    elif loss_type == "hinge":
        losses = nn.relu(1 - beta * logits)
    elif loss_type == "ipo":
        losses = (logits - 1 / (2 * beta)) ** 2
    elif loss_type == "dpop":
        delta = 50
        penalty = mx.maximum(mx.zeros_like(policy_chosen_score), reference_chosen_score - policy_chosen_score)
        losses = -(nn.log_sigmoid(beta * logits) - delta * penalty)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    loss = mx.mean(losses)
    num_tokens = (num_chosen_tokens + num_rejected_tokens).sum()

    chosen_reward = beta * mx.mean(policy_chosen_score - reference_chosen_score)
    rejected_reward = beta * mx.mean(policy_rejected_score - reference_rejected_score)
    reward = mx.stack([chosen_reward, rejected_reward])

    return loss, reward, num_tokens


def iterate_dpo_batches(dataset, batch_size, max_seq_length, train=False):
    idx = sorted(range(len(dataset)), key=lambda idx: len(dataset[idx]['chosen']))
    
    step = mx.distributed.init().size()
    if batch_size % step != 0:
        raise ValueError("Batch size must be divisible by workers")
        
    batch_idx = [idx[i:i+batch_size:step] for i in range(0, len(idx)-batch_size+1, batch_size)]
    
    while True:
        indices = np.random.permutation(len(batch_idx)) if train else range(len(batch_idx))
        for i in indices:
            batch = [dataset[j] for j in batch_idx[i]]
            
            # Get and process lengths
            chosen_lengths = [len(x['chosen']) for x in batch]
            rejected_lengths = [len(x['rejected']) for x in batch]
            max_length = min(max(max(chosen_lengths), max(rejected_lengths)), max_seq_length)
            
            # Dynamic padding based on batch content
            max_length_in_batch = max_length
            
            chosen_arr = np.zeros((batch_size // step, max_length_in_batch), np.int32)
            rejected_arr = np.zeros((batch_size // step, max_length_in_batch), np.int32)
            
            chosen_masks = np.zeros((batch_size // step, max_length_in_batch), np.float32)
            rejected_masks = np.zeros((batch_size // step, max_length_in_batch), np.float32)
                    
            for j in range(batch_size // step):
                chosen_length = min(chosen_lengths[j], max_seq_length)
                rejected_length = min(rejected_lengths[j], max_seq_length)
                        
                chosen_arr[j, :chosen_length] = batch[j]['chosen'][:chosen_length]
                rejected_arr[j, :rejected_length] = batch[j]['rejected'][:rejected_length]
                
                chosen_masks[j, :chosen_length] = 1.0
                rejected_masks[j, :rejected_length] = 1.0
                    
            yield mx.array(chosen_arr), mx.array(rejected_arr), mx.array(chosen_masks), mx.array(rejected_masks)
            
        if not train:
            break


def evaluate_dpo(
    model,
    reference_model,
    dataset,
    tokenizer,
    batch_size,
    num_batches,
    beta: float,
    delta: float,
    max_seq_length=2048,
    loss_fn: callable = dpo_loss,
    loss_type="sigmoid",
):
    all_losses = 0
    all_rewards = mx.zeros((2,))  # [chosen_reward, rejected_reward]
    ntokens = 0

    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)

    for _, batch in zip(
        index_iterator,
        iterate_dpo_batches(
            dataset=dataset,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        ),
    ):
        chosen, rejected, chosen_masks, rejected_masks = batch
        loss, reward, toks = loss_fn(
            model=model,
            reference_teacher_model=reference_model,
            chosen=chosen,
            rejected=rejected,
            chosen_masks=chosen_masks,
            rejected_masks=rejected_masks,
            loss_type=loss_type,
            beta=beta,
            delta=delta,
        )
        
        all_losses += loss * toks
        all_rewards += reward
        ntokens += toks
        mx.eval(all_losses, all_rewards, ntokens)

    all_losses = mx.distributed.all_sum(all_losses)
    all_rewards = mx.distributed.all_sum(all_rewards)
    ntokens = mx.distributed.all_sum(ntokens)

    return (all_losses / ntokens).item(), all_rewards.tolist()

def train_dpo(
    model,
    reference_model,
    tokenizer,
    optimizer,
    train_dataset,
    val_dataset,
    args: DPOTrainingArgs = DPOTrainingArgs(),
    loss_fn: callable = dpo_loss,
    training_callback: TrainingCallback = None,
    loss_type="sigmoid",
):
    print(f"Starting DPO training..., iters: {args.iters}")
    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()
    if world_size > 1:
        print(f"Node {rank} of {world_size}")

    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    state = [model.state, optimizer.state]

    def step(batch):
        chosen, rejected, chosen_masks, rejected_masks = batch
        
        # Remove loss_type from the call
        (loss, reward, toks), grad = loss_value_and_grad(
            model, 
            reference_model, 
            chosen, 
            rejected, 
            chosen_masks, 
            rejected_masks
        )

        # All reduce the gradients if running in distributed mode
        grad = average_gradients(grad)

        # Model update
        optimizer.update(model, grad)

        return loss, reward, toks

    # Create a wrapper function that includes all required arguments
    def loss_wrapper(model, ref_model, chosen, rejected, chosen_masks, rejected_masks):
        return loss_fn(
            model=model,
            reference_teacher_model=ref_model,
            chosen=chosen,
            rejected=rejected,
            chosen_masks=chosen_masks,
            rejected_masks=rejected_masks,
            beta=args.beta,
            delta=args.delta,
            loss_type=loss_type,
            is_reference_free=args.is_reference_free
        )
    
    # Create value_and_grad with the wrapper
    loss_value_and_grad = nn.value_and_grad(model, loss_wrapper)

    losses = 0
    rewards = mx.zeros((2,))
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    
    # Main training loop
    start = time.perf_counter()
    for it, batch in zip(
        range(1, args.iters + 1),
        iterate_dpo_batches(
            dataset=train_dataset,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            train=True,
        ),
    ):
        # Report validation loss if needed
        if it == 1 or it % args.steps_per_eval == 0 or it == args.iters:
            stop = time.perf_counter()
            val_loss, val_rewards = evaluate_dpo(
                model=model,
                reference_model=reference_model,
                dataset=val_dataset,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                loss_fn=loss_fn,
                beta=args.beta,
                delta=args.delta,
                loss_type=loss_type,
            )
            val_time = time.perf_counter() - stop
            if rank == 0:
                print(
                    f"Iter {it}: "
                    f"Val loss {val_loss:.8f}, "
                    f"Val chosen reward {val_rewards[0]:.3f}, "
                    f"Val rejected reward {val_rewards[1]:.3f}, "
                    f"Val took {val_time:.3f}s",
                    flush=True,
                )

            if training_callback is not None:
                val_info = {
                    "iteration": it,
                    "val_loss": val_loss,
                    "val_chosen_reward": val_rewards[0],
                    "val_rejected_reward": val_rewards[1],
                    "val_time": val_time,
                }
                training_callback.on_val_loss_report(val_info)

            start = time.perf_counter()

        loss, reward, toks = step(batch)
        losses += loss
        rewards += reward
        n_tokens += toks
        steps += 1
        mx.eval(state, losses, rewards, n_tokens)

        # Report training loss if needed
        if it % args.steps_per_report == 0 or it == args.iters:
            stop = time.perf_counter()

            train_loss = mx.distributed.all_sum(losses).item()
            train_loss /= steps * world_size
            train_rewards = mx.distributed.all_sum(rewards).tolist()
            train_rewards = [r / (steps * world_size) for r in train_rewards]
            n_tokens = mx.distributed.all_sum(n_tokens).item()
            learning_rate = optimizer.learning_rate.item()
            it_sec = args.steps_per_report / (stop - start)
            tokens_sec = float(n_tokens) / (stop - start)
            trained_tokens += n_tokens
            peak_mem = mx.metal.get_peak_memory() / 1e9
            
            if rank == 0:
                print(
                    f"Iter {it}: Train loss {train_loss:.8f}, "
                    f"Chosen reward {train_rewards[0]:.3f}, "
                    f"Rejected reward {train_rewards[1]:.3f}, "
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
                    "train_chosen_reward": train_rewards[0],
                    "train_rejected_reward": train_rewards[1],
                    "learning_rate": learning_rate,
                    "iterations_per_second": it_sec,
                    "tokens_per_second": tokens_sec,
                    "trained_tokens": trained_tokens,
                    "peak_memory": peak_mem,
                }
                training_callback.on_train_loss_report(train_info)

            losses = 0
            rewards = mx.zeros((2,))
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