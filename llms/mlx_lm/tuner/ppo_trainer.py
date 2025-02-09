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


@dataclass
class PPOTrainingArgs(TrainingArgs):
    vf_coef: float = field(default=0.5, metadata={"help": "Value function coefficient"})
    cliprange: float = field(default=0.2, metadata={"help": "Policy gradient clipping range"}) 
    cliprange_value: float = field(default=0.2, metadata={"help": "Value function clipping range"})
    gamma: float = field(default=0.99, metadata={"help": "Discount factor"})
    lambda_: float = field(default=0.95, metadata={"help": "GAE lambda"})


def compute_returns(
    rewards: mx.array, 
    gamma: float = 0.99
) -> mx.array:
   """Compute returns with Generalized Advantage Estimation"""
   returns = mx.zeros_like(rewards)
   running_return = 0
   
   for t in reversed(range(len(rewards))):
       running_return = rewards[t] + gamma * running_return 
       returns = returns.at[t].set(running_return)
       
   return returns

def compute_advantages(
    values: mx.array,
    returns: mx.array,
    rewards: mx.array,
    gamma: float = 0.99,
    lambda_: float = 0.95
) -> mx.array:
   """Compute advantages using GAE"""
   advantages = mx.zeros_like(returns)
   running_advantage = 0
   
   for t in reversed(range(len(returns))):
       if t < len(returns) - 1:
           delta = rewards[t] + gamma * values[t + 1] - values[t]
       else:
           delta = rewards[t] - values[t]
           
       running_advantage = delta + gamma * lambda_ * running_advantage
       advantages = advantages.at[t].set(running_advantage)
       
   return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

def make_predictions(model, x, mask):
    inputs = x[:, :-1]
    targets = x[:, 1:]
    
    logits = model(inputs)
    logits = logits.astype(mx.float32)

    return -nn.losses.cross_entropy(logits, targets) * mask[:, :-1]

def compute_rewards(model, x, mask, reward_scale=1.0):
   """
   Compute rewards based on model predictions and actual targets.
   Basic implementation using log probabilities as rewards.
   """
   logits = model(x[:, :-1])
   targets = x[:, 1:]
   
   log_probs = -nn.losses.cross_entropy(logits, targets, reduction='none')
   rewards = log_probs * mask[:, :-1] * reward_scale
   
   return rewards
   
def ppo_loss(
    model,
    inputs,
    mask,
    old_logprobs,
    values,
    old_values,
    advantages,
    returns,
    vf_coef=0.5,
    cliprange=0.2,
    cliprange_value=0.2
): 
   # Get new log probs
   new_logprobs = make_predictions(model, inputs, mask)
   ntoks = mask[:, :-1].sum()
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


def iterate_ppo_batches(dataset, tokenizer, batch_size, max_seq_length, train=False):
   # Sort by length
   idx = sorted(range(len(dataset)), key=lambda idx: len(dataset[idx]))
   if len(dataset) < batch_size:
       raise ValueError(f"Dataset must have at least batch_size={batch_size} examples but only has {len(dataset)}.")
   
   # Handle distributed training
   step = mx.distributed.init().size()
   if batch_size % step != 0:
       raise ValueError("The batch size must be divisible by the number of workers")
   
   # Make batches
   batch_idx = [idx[i:i+batch_size:step] for i in range(0, len(idx)-batch_size+1, batch_size)]
   
   while True:
       indices = np.random.permutation(len(batch_idx))
       for i in indices:
           batch = [dataset[j] for j in batch_idx[i]]
           lengths = [len(x) for x in batch]
           
           # Handle sequence length
           if max(lengths) > max_seq_length:
               print(f"[WARNING] Truncating sequences longer than {max_seq_length}")
           
           # Pad to multiple of 8
           pad_to = 8
           max_length_in_batch = pad_to * ((max(lengths) + pad_to - 1) // pad_to)
           max_length_in_batch = min(max_length_in_batch, max_seq_length)
           
           # Create batch array
           batch_arr = np.zeros((batch_size // step, max_length_in_batch), np.int32)
           mask = np.zeros((batch_size // step, max_length_in_batch), np.int32)
           
           for j in range(batch_size // step):
               truncated_length = min(lengths[j], max_seq_length)
               batch_arr[j, :truncated_length] = batch[j][:truncated_length]
               mask[j, :truncated_length] = 1
               lengths[j] = truncated_length

           batch = mx.array(batch_arr)
           mask = mx.array(mask)
           
           yield batch, mask
           
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
   loss: callable = ppo_loss,
   iterate_ppo_batches: callable = iterate_ppo_batches,
):
   total_loss = 0
   total_pg_loss = 0 
   total_vf_loss = 0
   ntokens = 0

   index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)

   for _, batch in zip(
       index_iterator,
       iterate_ppo_batches(
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
    args: PPOTrainingArgs = PPOTrainingArgs(),
    loss: callable = ppo_loss,
    iterate_ppo_batches: callable = iterate_ppo_batches,
    training_callback: TrainingCallback = None,
):
    print(f"Starting PPO training..., iters: {args.iters}")
    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()
    if world_size > 1:
        print(f"Node {rank} of {world_size}")

    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    state = [model.state, optimizer.state]

    def step(batch):
        x, mask = batch
        
        # Initial forward pass
        old_logprobs = make_predictions(model, x, mask)
        values = model.value_head(x[:, :-1])
        old_values = values.copy()
        
        # Compute rewards (implement reward calculation based on your task)
        rewards = compute_rewards(model, x, mask)
        
        # Compute returns and advantages
        returns = compute_returns(rewards, values, gamma=args.gamma)
        advantages = compute_advantages(values, returns, rewards, 
                                        gamma=args.gamma,
                                        lambda_=args.lambda_)
        
        def loss_fn(model, x, mask):
            total_loss, pg_loss, vf_loss, ntoks = ppo_loss(
                model, x, mask,
                old_logprobs, values, old_values,
                advantages, returns,
                vf_coef=args.vf_coef,
                cliprange=args.cliprange, 
                cliprange_value=args.cliprange_value
            )
            return total_loss, ntoks, pg_loss, vf_loss
        
        (loss_val, toks, pg_loss, vf_loss), grad = nn.value_and_grad(model, loss_fn)(x, mask)
        grad = average_gradients(grad)
        optimizer.update(model, grad)
        
        return loss_val, toks, pg_loss, vf_loss

    losses = 0
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    # Main training loop
    start = time.perf_counter()
    for it, batch in zip(
        range(1, args.iters + 1),
        iterate_ppo_batches(
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
                iterate_ppo_batches=iterate_ppo_batches,
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
