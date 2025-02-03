# Copyright © 2024 Apple Inc.

import time
from dataclasses import dataclass, field
from pathlib import Path
import re

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from .trainer import grad_checkpoint, TrainingArgs, TrainingCallback, average_gradients, iterate_batches

from mlx_lm import generate


@dataclass
class GRPOTrainingArgs(TrainingArgs):
    group_size: int = field(
        default=4,
        metadata={"help": "Number of responses per prompt."},
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


def r1_extract_xml_answer(text: str) -> str:
    """Extracts the answer from an XML formatted text string."""
    try:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    except:
        print("[extract_xml_answer] Failed to extract answer from: ", text)
        return ""

def r1_accuracy_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Calculates reward based on accuracy of extracted answers.
    
    Args:
        prompts: List of input prompts
        completions: List of completion strings
        answer: Expected answer or list of answers
        **kwargs: Additional arguments
        
    Returns:
        list[float]: Reward values for each completion
    """
    extracted_responses = [r1_extract_xml_answer(r) for r in completions]
    q = prompts[0] if isinstance(prompts[0], str) else prompts[0][-1]['content']
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def r1_int_reward_func(completions, **kwargs) -> list[float]:
    """Rewards numerical responses.
    
    Args:
        completions: List of completion strings
        **kwargs: Additional arguments
        
    Returns:
        list[float]: Reward values for each completion
    """
    extracted_responses = [r1_extract_xml_answer(r) for r in completions]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def r1_strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Rewards completions with strict XML format.
    
    Args:
        completions: List of completion strings
        **kwargs: Additional arguments
        
    Returns:
        list[float]: Reward values for each completion
    """
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    matches = [re.match(pattern, r) for r in completions]
    return [0.5 if match else 0.0 for match in matches]

def r1_soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Rewards completions with flexible XML format.
    
    Args:
        completions: List of completion strings
        **kwargs: Additional arguments
        
    Returns:
        list[float]: Reward values for each completion
    """
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    matches = [re.match(pattern, r) for r in completions]
    return [0.5 if match else 0.0 for match in matches]

def r1_count_xml(text: str) -> float:
    """Calculates score based on XML formatting.
    
    Args:
        text: Input text string
        
    Returns:
        float: Score based on XML tag presence and formatting
    """
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count


def grpo_loss(
        model,
        tokenizer,
        prompts,
        reward_funcs=None,
        beta=0.1,
        group_size=4,
        epsilon=1e-4,
        ref_model=None
    ):
    """
    Calculates the GRPO loss with support for multiple reward functions.
    
    Args:
        model: The model to optimize
        tokenizer: Tokenizer for processing inputs
        prompts: List of input prompts
        reward_funcs: List of reward functions to use
        beta: KL penalty coefficient
        group_size: Number of completions per prompt
        epsilon: Small constant for numerical stability
        ref_model: Optional reference model for KL divergence
        
    Returns:
        tuple: (loss, total_sequence_length, metrics_dict)
    """
    batch_size = len(prompts)
    # Generate multiple completions for each prompt
    all_completions = []
    
    for prompt in prompts:
        prompt_completions = []
        for _ in range(group_size):
            completion = generate(model, tokenizer, prompt)
            prompt_completions.append(completion)
        all_completions.extend(prompt_completions)
    
    # Tokenize all prompts + completions
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
    
    # Prepare targets
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
    
    # Compute KL divergence
    kl_div = (mx.exp(ref_token_log_probs - token_log_probs) - (ref_token_log_probs - token_log_probs) - 1)

    # Calculate combined rewards from all reward functions
    rewards = mx.zeros((len(all_completions),))
    for reward_func in reward_funcs:
        func_rewards = mx.array(reward_func(all_completions))
        rewards += func_rewards

    # Normalize rewards if using multiple reward functions
    if len(reward_funcs) > 1:
        rewards /= len(reward_funcs)

    # Compute grouped-wise rewards
    grouped_rewards = rewards.reshape(batch_size, group_size)
    mean_grouped_rewards = mx.mean(grouped_rewards, axis=1)
    std_grouped_rewards = mx.std(grouped_rewards, axis=1)

    # Normalize rewards to compute advantages
    mean_grouped_rewards = mx.repeat(mean_grouped_rewards.reshape(-1, 1), group_size, axis=1).reshape(-1)
    std_grouped_rewards = mx.repeat(std_grouped_rewards.reshape(-1, 1), group_size, axis=1).reshape(-1)
    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + epsilon)

    # Create length mask for the shifted sequence
    length_mask = mx.arange(inputs.shape[1] - 1)[None, :] < (lengths[:, None] - 1)

    # Calculate policy gradient loss
    per_token_loss = mx.exp(token_log_probs - mx.stop_gradient(token_log_probs)) * advantages.reshape(-1, 1)
    per_token_loss = -(per_token_loss - beta * kl_div)

    # Normalize loss properly per sequence
    sequence_sums = (per_token_loss * length_mask).sum(axis=1)
    sequence_lengths = length_mask.sum(axis=1)
    loss = (sequence_sums / sequence_lengths).mean()

    # Calculate mean KL divergence
    mean_kl = ((kl_div * length_mask).sum(axis=1) / length_mask.sum(axis=1)).mean()
    
    # Collect metrics for each reward function separately
    reward_metrics = {}
    for i, reward_func in enumerate(reward_funcs):
        func_rewards = mx.array(reward_func(all_completions))
        func_grouped_rewards = func_rewards.reshape(batch_size, group_size)
        reward_metrics[f'reward_func_{i}_mean'] = mx.mean(func_rewards)
        reward_metrics[f'reward_func_{i}_std'] = mx.std(func_rewards)

    metrics = {
        'total_rewards_mean': mx.mean(rewards),
        'total_rewards_std': mx.std(rewards),
        'grouped_rewards_mean': mx.mean(grouped_rewards),
        'grouped_rewards_std': mx.std(grouped_rewards),
        'kl': mean_kl,
        **reward_metrics
    }

    return loss, sequence_lengths.sum(), metrics


def iterate_grpo_batches(dataset, tokenizer, batch_size, max_seq_length, train=False):
    """
    Creates batches from prompt-answer pairs for GRPO training.
    
    Args:
        dataset: List of (prompt, answer) pairs
        tokenizer: Tokenizer for processing inputs
        batch_size: Size of each batch
        max_seq_length: Maximum sequence length
        train: Whether this is for training
        
    Yields:
        List of prompts for the current batch
    """
    # Verify dataset is not empty and has correct format
    if not dataset or not isinstance(dataset[0], (tuple, list)) or len(dataset[0]) != 2:
        raise ValueError("Dataset must be a list of (prompt, answer) pairs")
    
    # Sort by combined length of prompt + answer
    idx = sorted(range(len(dataset)), 
                key=lambda i: len(dataset[i][0]) + len(dataset[i][1]))
    
    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size} "
            f"examples but only has {len(dataset)}."
        )

    # Handle distributed training
    step = mx.distributed.init().size()
    if batch_size % step != 0:
        raise ValueError("The batch size must be divisible by the number of workers")

    # Create batch indices
    batch_idx = [
        idx[i : i + batch_size : step]
        for i in range(0, len(idx) - batch_size + 1, batch_size)
    ]

    while True:
        # Shuffle batch indices if training
        indices = np.random.permutation(len(batch_idx)) if train else range(len(batch_idx))
        
        for i in indices:
            # Get current batch of prompt-answer pairs
            current_batch = [dataset[j] for j in batch_idx[i]]
            
            # Extract prompts and answers
            prompts = [pair[0] for pair in current_batch]
            answers = [pair[1] for pair in current_batch]
            
            if any(len(p) > max_seq_length for p in prompts):
                print(
                    f"[WARNING] Some prompts are longer than {max_seq_length} tokens. "
                    "Long prompts will be truncated."
                )
            
            # For GRPO, we only need to yield the prompts
            # The answers will be used by the reward functions
            yield prompts

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
    epsilon: float,
    group_size: int,
    max_seq_length,
    reward_funcs = None,
    loss: callable = grpo_loss,
    iterate_batches: callable = iterate_grpo_batches
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
            epsilon=epsilon,
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
    ref_model,
    tokenizer,
    optimizer,
    train_dataset,
    val_dataset,
    reward_funcs = [
        r1_accuracy_reward_func,
        r1_int_reward_func,
        r1_strict_format_reward_func,
        r1_soft_format_reward_func,
        r1_count_xml
    ],
    args: GRPOTrainingArgs = GRPOTrainingArgs(),
    loss: callable = grpo_loss,
    iterate_batches: callable = iterate_grpo_batches,
    training_callback: TrainingCallback = None,
):
    print(f"Starting GRPO training with {len(reward_funcs)} reward functions..., iters: {args.iters}")
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
    for i in range(len(reward_funcs)):
        accumulated_metrics[f'reward_func_{i}_mean'] = 0
        accumulated_metrics[f'reward_func_{i}_std'] = 0

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
                ref_model=ref_model,
                reward_funcs=reward_funcs,
                tokenizer=tokenizer,
                group_size=args.group_size,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                beta=args.beta,
                epsilon=args.epsilon,
                iterate_batches=iterate_batches,
            )
            val_time = time.perf_counter() - stop
            if rank == 0:
                val_metrics_str = (
                    f"Val loss {val_loss:.8f}, "
                    f"Val total_rewards_mean {val_metrics['total_rewards_mean']:.3f}, "
                    f"Val total_rewards_std {val_metrics['total_rewards_std']:.3f}, "
                    f"Val grouped_rewards_mean {val_metrics['grouped_rewards_mean']:.3f}, "
                    f"Val grouped_rewards_std {val_metrics['grouped_rewards_std']:.3f}, "
                    f"Val kl {val_metrics['kl']:.3f}"
                )
                
                # Add reward function specific metrics
                for i in range(len(reward_funcs)):
                    val_metrics_str += (
                        f", Val reward_func_{i}_mean {val_metrics[f'reward_func_{i}_mean']:.3f}, "
                        f"Val reward_func_{i}_std {val_metrics[f'reward_func_{i}_std']:.3f}"
                    )
                
                print(
                    f"Iter {it}: {val_metrics_str}, "
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
                train_metrics_str = (
                    f"Train loss {train_loss:.8f}, "
                    f"Total rewards mean {avg_metrics['total_rewards_mean']:.3f}, "
                    f"Total rewards std {avg_metrics['total_rewards_std']:.3f}, "
                    f"Grouped rewards mean {avg_metrics['grouped_rewards_mean']:.3f}, "
                    f"Grouped rewards std {avg_metrics['grouped_rewards_std']:.3f}, "
                    f"KL {avg_metrics['kl']:.3f}"
                )
                
                # Add reward function specific metrics
                for i in range(len(reward_funcs)):
                    train_metrics_str += (
                        f", Reward func {i} mean {avg_metrics[f'reward_func_{i}_mean']:.3f}, "
                        f"Reward func {i} std {avg_metrics[f'reward_func_{i}_std']:.3f}"
                    )

                print(
                    f"Iter {it}: {train_metrics_str}, "
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