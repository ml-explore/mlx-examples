import time
from pathlib import Path
from dataclasses import dataclass, field

import mlx.nn as nn
import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx.nn.utils import average_gradients
from .trainer import TrainingArgs, grad_checkpoint, TrainingCallback


@dataclass
class ORPOTrainingArgs(TrainingArgs):
    beta: float = field(
        default=0.1,
        metadata={"help": "Temperature parameter for DPO training."}
    )
    reward_scaling: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for offline rewards."}
    )


def orpo_loss(
    model,
    chosen: mx.array,
    rejected: mx.array,
    chosen_masks: mx.array,
    rejected_masks: mx.array,
    chosen_rewards: mx.array,
    rejected_rewards: mx.array,
    beta: float,
    reward_scaling: float = 1.0,
):
    """
    Calculate ORPO loss using pre-computed rewards.
    Args:
        model: Policy model
        chosen: Chosen sequence tokens
        rejected: Rejected sequence tokens
        chosen_masks: Attention masks for chosen sequences
        rejected_masks: Attention masks for rejected sequences
        chosen_rewards: Pre-computed rewards for chosen sequences
        rejected_rewards: Pre-computed rewards for rejected sequences
        beta: Temperature parameter
        reward_scaling: Scaling factor for rewards
    Returns:
        Loss value, rewards, and number of tokens.
    """
    def make_predictions(model, x, mask):
        inputs = x[:, :-1]
        targets = x[:, 1:]
        
        logits = model(inputs)
        logits = logits.astype(mx.float32)
        
        return -nn.losses.cross_entropy(logits, targets) * mask[:, :-1]

    # Calculate log probabilities for policy model
    policy_chosen_scores = make_predictions(model, chosen, chosen_masks)
    policy_rejected_scores = make_predictions(model, rejected, rejected_masks)
    
    # Scale the pre-computed rewards
    chosen_rewards = chosen_rewards * reward_scaling
    rejected_rewards = rejected_rewards * reward_scaling
    
    # ORPO uses the reward difference directly
    reward_diff = chosen_rewards - rejected_rewards
    
    # Calculate ORPO loss using logistic function
    policy_diff = policy_chosen_scores.sum(-1) - policy_rejected_scores.sum(-1)
    loss = -nn.log_sigmoid(beta * (policy_diff * reward_diff))
    
    loss = mx.mean(loss)
    
    # Calculate number of tokens for logging
    num_tokens = (chosen_masks.sum() + rejected_masks.sum())
    
    # Calculate rewards for logging
    avg_chosen_reward = mx.mean(chosen_rewards)
    avg_rejected_reward = mx.mean(rejected_rewards)
    reward = mx.stack([avg_chosen_reward, avg_rejected_reward])

    return loss, reward, num_tokens


def evaluate_orpo(
    model,
    dataset,
    tokenizer,
    batch_size,
    num_batches,
    beta: float,
    reward_scaling: float = 1.0,
    max_seq_length=2048,
):
    """
    Evaluation function for ORPO.
    """
    all_losses = 0
    all_rewards = mx.zeros((2,))
    ntokens = 0

    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)

    for _, batch in zip(
        index_iterator,
        iterate_orpo_batches(
            dataset=dataset,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        ),
    ):
        chosen, rejected, chosen_masks, rejected_masks, chosen_rewards, rejected_rewards = batch
        loss, reward, toks = orpo_loss(
            model=model,
            chosen=chosen,
            rejected=rejected,
            chosen_masks=chosen_masks,
            rejected_masks=rejected_masks,
            chosen_rewards=chosen_rewards,
            rejected_rewards=rejected_rewards,
            beta=beta,
            reward_scaling=reward_scaling,
        )
        
        all_losses += loss * toks
        all_rewards += reward
        ntokens += toks
        mx.eval(all_losses, all_rewards, ntokens)

    all_losses = mx.distributed.all_sum(all_losses)
    all_rewards = mx.distributed.all_sum(all_rewards)
    ntokens = mx.distributed.all_sum(ntokens)

    return (all_losses / ntokens).item(), all_rewards.tolist()


def iterate_orpo_batches(dataset, tokenizer, batch_size, max_seq_length, train=False):
    """
    Modified batch iterator for ORPO that includes pre-computed rewards.
    Works with pre-tokenized input data.
    """
    # Sort pairs by length of the chosen response
    idx = sorted(range(len(dataset)), key=lambda idx: len(dataset[idx]['chosen']))
    
    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size}"
            f" examples but only has {len(dataset)}."
        )

    step = mx.distributed.init().size()
    if batch_size % step != 0:
        raise ValueError("The batch size must be divisible by the number of workers")

    batch_idx = [
        idx[i : i + batch_size : step]
        for i in range(0, len(idx) - batch_size + 1, batch_size)
    ]

    while True:
        indices = np.random.permutation(len(batch_idx)) if train else range(len(batch_idx))
        for i in indices:
            batch = [dataset[j] for j in batch_idx[i]]
            
            # Get lengths assuming data is already tokenized
            chosen_lengths = [len(x['chosen']) for x in batch]
            rejected_lengths = [len(x['rejected']) for x in batch]
            max_length = max(max(chosen_lengths), max(rejected_lengths))
            
            if max_length > max_seq_length:
                print(
                    f"[WARNING] Sequences longer than {max_seq_length} tokens "
                    f"will be truncated."
                )

            pad_to = 8
            max_length_in_batch = pad_to * ((max_length + pad_to - 1) // pad_to)
            max_length_in_batch = min(max_length_in_batch, max_seq_length)

            chosen_arr = np.zeros((batch_size // step, max_length_in_batch), np.int32)
            rejected_arr = np.zeros((batch_size // step, max_length_in_batch), np.int32)
            chosen_masks = np.zeros((batch_size // step, max_length_in_batch), np.float32)
            rejected_masks = np.zeros((batch_size // step, max_length_in_batch), np.float32)
            
            # Always use binary rewards
            chosen_rewards = np.ones((batch_size // step,), np.float32)
            rejected_rewards = np.zeros((batch_size // step,), np.float32)

            for j in range(batch_size // step):
                # Use pre-tokenized sequences directly
                chosen_length = min(chosen_lengths[j], max_seq_length)
                chosen_arr[j, :chosen_length] = batch[j]['chosen'][:chosen_length]
                chosen_masks[j, :chosen_length] = 1.0

                rejected_length = min(rejected_lengths[j], max_seq_length)
                rejected_arr[j, :rejected_length] = batch[j]['rejected'][:rejected_length]
                rejected_masks[j, :rejected_length] = 1.0

            yield (mx.array(chosen_arr), mx.array(rejected_arr),
                  mx.array(chosen_masks), mx.array(rejected_masks),
                  mx.array(chosen_rewards), mx.array(rejected_rewards))

        if not train:
            break


def train_orpo(
    model,
    tokenizer,
    optimizer,
    train_dataset,
    val_dataset,
    args: ORPOTrainingArgs = ORPOTrainingArgs(),
    training_callback: TrainingCallback = None,
):
    """
    Training function for ORPO.
    """
    print(f"Starting ORPO training..., iters: {args.iters}")
    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()
    
    if world_size > 1:
        print(f"Node {rank} of {world_size}")

    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    state = [model.state, optimizer.state]

    def step(batch):
        chosen, rejected, chosen_masks, rejected_masks, chosen_rewards, rejected_rewards = batch
        
        (loss, reward, toks), grad = loss_value_and_grad(
            model, 
            chosen, 
            rejected, 
            chosen_masks, 
            rejected_masks,
            chosen_rewards,
            rejected_rewards
        )

        grad = average_gradients(grad)
        optimizer.update(model, grad)

        return loss, reward, toks

    def loss_wrapper(model, chosen, rejected, chosen_masks, rejected_masks, 
                    chosen_rewards, rejected_rewards):
        return orpo_loss(
            model=model,
            chosen=chosen,
            rejected=rejected,
            chosen_masks=chosen_masks,
            rejected_masks=rejected_masks,
            chosen_rewards=chosen_rewards,
            rejected_rewards=rejected_rewards,
            beta=args.beta,
            reward_scaling=args.reward_scaling
        )
    
    loss_value_and_grad = nn.value_and_grad(model, loss_wrapper)

    # Training loop with progress tracking
    losses = 0
    rewards = mx.zeros((2,))
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    
    start = time.perf_counter()
    for it, batch in zip(
        range(1, args.iters + 1),
        iterate_orpo_batches(  # reuse DPO batch iterator
            dataset=train_dataset,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            train=True,
        ),
    ):
        # Evaluate if needed
        if it == 1 or it % args.steps_per_eval == 0 or it == args.iters:
            stop = time.perf_counter()
            val_loss, val_rewards = evaluate_orpo(
                model=model,
                dataset=val_dataset,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                beta=args.beta,
                reward_scaling=args.reward_scaling,
            )
            val_time = time.perf_counter() - stop
            if rank == 0:
                print(
                    f"Iter {it}: "
                    f"Val loss {val_loss:.3f}, "
                    f"Val chosen reward {val_rewards[0]:.3f}, "
                    f"Val rejected reward {val_rewards[1]:.3f}, "
                    f"Val took {val_time:.3f}s",
                    flush=True,
                )

            if training_callback is not None:
                training_callback.on_val_loss_report({
                    "iteration": it,
                    "val_loss": val_loss,
                    "val_chosen_reward": val_rewards[0],
                    "val_rejected_reward": val_rewards[1],
                    "val_time": val_time,
                })

            start = time.perf_counter()

        # Training step
        loss, reward, toks = step(batch)
        losses += loss
        rewards += reward
        n_tokens += toks
        steps += 1
        mx.eval(state, losses, rewards, n_tokens)

        # Report training metrics if needed
        if it % args.steps_per_report == 0 or it == args.iters:
            stop = time.perf_counter()

            train_loss = mx.distributed.all_sum(losses).item() / (steps * world_size)
            train_rewards = [r / (steps * world_size) for r in mx.distributed.all_sum(rewards).tolist()]
            n_tokens = mx.distributed.all_sum(n_tokens).item()
            learning_rate = optimizer.learning_rate.item()
            it_sec = args.steps_per_report / (stop - start)
            tokens_sec = float(n_tokens) / (stop - start)
            trained_tokens += n_tokens
            peak_mem = mx.metal.get_peak_memory() / 1e9
            
            if rank == 0:
                print(
                    f"Iter {it}: Train loss {train_loss:.3f}, "
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
                training_callback.on_train_loss_report({
                    "iteration": it,
                    "train_loss": train_loss,
                    "train_chosen_reward": train_rewards[0],
                    "train_rejected_reward": train_rewards[1],
                    "learning_rate": learning_rate,
                    "iterations_per_second": it_sec,
                    "tokens_per_second": tokens_sec,
                    "trained_tokens": trained_tokens,
                    "peak_memory": peak_mem,
                })

            losses = 0
            rewards = mx.zeros((2,))
            n_tokens = 0
            steps = 0
            start = time.perf_counter()

        # Save model weights if needed
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