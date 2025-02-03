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

from mlx_lm.utils import generate_step


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


def generate_grpo(model, prompt, max_tokens, tokenizer, temperature=1.0):
    model.eval()
    if len(prompt.shape) == 1:
        prompt = prompt[None, :]
    
    generated = []
    current_prompt = prompt[0]
    
    for _ in range(max_tokens):
        current_batch = current_prompt[None, :]
        logits = model(current_batch)
        token_logits = logits[0, -1]
        
        if temperature > 0:
            token_logits = token_logits / temperature
            
        probs = mx.softmax(token_logits)
        next_token = mx.random.categorical(probs[None, :])
        next_token = next_token[0]
        mx.eval(next_token)
        
        token_value = next_token.item()
        generated.append(next_token)
        
        # Clear intermediate tensors
        del logits, token_logits, probs
        mx.metal.clear_cache()
        
        current_prompt = mx.concatenate([current_prompt, next_token[None]])
        if token_value == tokenizer.eos_token_id:
            break
            
    if not generated:
        return prompt[0]
        
    result = mx.concatenate([prompt[0], mx.stack(generated)])
    mx.eval(result)
    model.train()
    
    # Clear generated tokens
    del generated
    mx.metal.clear_cache()
    
    return result


def r1_extract_xml_answer(text: str) -> str:
    """Extracts the answer from an XML formatted text string."""
    try:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    except:
        print("[extract_xml_answer] Failed to extract answer from: ", text)
        return ""

def r1_accuracy_reward_func(prompts: list, completions: list, answer: list, **kwargs) -> list[float]:
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
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def r1_int_reward_func(prompts: list, completions: list, answer: list, **kwargs) -> list[float]:
    """Rewards numerical responses.
    Args:
        prompts: List of input prompts
        completions: List of completion strings
        answer: Expected answer or list of answers
        **kwargs: Additional arguments
    Returns:
        list[float]: Reward values for each completion
    """
    extracted_responses = [r1_extract_xml_answer(r) for r in completions]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def r1_soft_format_reward_func(prompts: list, completions: list, answer: list, **kwargs) -> list[float]:
    """Rewards completions with flexible XML format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    matches = [re.match(pattern, r) for r in completions]
    return [0.5 if match else 0.0 for match in matches]

def r1_strict_format_reward_func(prompts: list, completions: list, answer: list, **kwargs) -> list[float]:
    """Rewards completions with strict XML format.
    Args:
        prompts: List of input prompts
        completions: List of completion strings
        answer: Expected answer or list of answers
        **kwargs: Additional arguments
    Returns:
        list[float]: Reward values for each completion
    """
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    matches = [re.match(pattern, r) for r in completions]
    return [0.5 if match else 0.0 for match in matches]

def r1_count_xml(prompts: list, completions: list, answer: list, **kwargs) -> list[float]:
    """Calculates score based on XML formatting.
    Args:
        prompts: List of input prompts (unused)
        completions: List of completion strings to evaluate
        answer: Expected answer or list of answers (unused)
        **kwargs: Additional arguments
    Returns:
        list[float]: List of scores based on XML tag presence and formatting
    """
    scores = []
    for text in completions:
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
        scores.append(count)
    return scores


def grpo_loss(
    model,
    tokenizer,
    batch,
    reward_funcs=None,
    beta=0.1,
    group_size=4,
    epsilon=1e-4,
    ref_model=None,
    max_tokens=64,
    temperature=1.0
):
    prompt_tokens, answer_tokens, prompt_text, answer_text = batch
    batch_size = len(prompt_tokens)
    
    # Generation logic remains the same
    all_completions = []
    all_completion_texts = []
    
    for prompt in prompt_tokens:
        prompt_tensor = mx.array(prompt)
        
        for _ in range(group_size):
            try:
                completion_ids = generate_grpo(model, prompt_tensor, max_tokens, tokenizer, temperature)
                if completion_ids is None:
                    continue
                    
                completion_text = tokenizer.decode(completion_ids.tolist())
                all_completions.append(completion_ids)
                all_completion_texts.append(completion_text)
                
                del completion_ids
                mx.metal.clear_cache()
                
            except Exception as e:
                print(f"Generation error: {e}")
                continue
            
        del prompt_tensor
        mx.metal.clear_cache()

    # Prepare inputs
    expanded_answers = []
    expanded_prompts = []
    for i in range(batch_size):
        expanded_answers.extend([answer_text[i]] * group_size)
        expanded_prompts.extend([prompt_text[i]] * group_size)
        
    max_length = max(ids.shape[0] for ids in all_completions)
    padded_completions = []
    attention_masks = []
    
    for completion_ids in all_completions:
        padding_length = max_length - completion_ids.shape[0]
        if padding_length > 0:
            padding = mx.zeros((padding_length,), dtype=completion_ids.dtype)
            padded_ids = mx.concatenate([completion_ids, padding])
            mask = mx.concatenate([mx.ones_like(completion_ids), mx.zeros_like(padding)])
        else:
            padded_ids = completion_ids
            mask = mx.ones_like(completion_ids)
        padded_completions.append(padded_ids)
        attention_masks.append(mask)
        
        del completion_ids
        if padding_length > 0:
            del padding
        del mask
        mx.metal.clear_cache()
    
    inputs = mx.stack(padded_completions)
    attention_mask = mx.stack(attention_masks)
    lengths = attention_mask.sum(axis=1)
    
    del padded_completions, attention_masks
    mx.metal.clear_cache()
    
    # Get logits and compute log probabilities
    logits = model(inputs).astype(mx.float32)
    log_probs = nn.log_softmax(logits[:, :-1, :], axis=-1)
    targets = inputs[:, 1:]
    
    # Current policy probabilities
    token_log_probs = mx.take_along_axis(
        log_probs,
        targets.reshape(*targets.shape, 1),
        axis=-1
    ).squeeze(-1)
    
    # Reference policy probabilities
    if ref_model is not None:
        ref_logits = ref_model(inputs).astype(mx.float32)
    else:
        ref_logits = mx.array(logits)
        
    ref_log_probs = nn.log_softmax(ref_logits[:, :-1, :], axis=-1)
    ref_token_log_probs = mx.take_along_axis(
        ref_log_probs,
        targets.reshape(*targets.shape, 1),
        axis=-1
    ).squeeze(-1)
    
    # Calculate rewards and advantages
    rewards = mx.zeros((len(all_completions),))
    for reward_func in reward_funcs:
        func_rewards = mx.array(reward_func(
            prompts=expanded_prompts,
            completions=all_completion_texts,
            answer=expanded_answers
        ))
        rewards += func_rewards

    if len(reward_funcs) > 1:
        rewards /= len(reward_funcs)

    # Reshape rewards and compute advantages following GRPO formula
    rewards_reshaped = rewards.reshape(batch_size, group_size)
    mean_rewards = mx.broadcast_to(mx.mean(rewards_reshaped, axis=1)[:, None], (rewards_reshaped.shape[0], group_size)).reshape(-1)
    std_rewards = mx.broadcast_to(mx.std(rewards_reshaped, axis=1)[:, None], (rewards_reshaped.shape[0], group_size)).reshape(-1)
    advantages = (rewards - mean_rewards) / (std_rewards + epsilon)
    
    # Compute KL divergence using Schulman's approximator
    kl_div = mx.exp(ref_token_log_probs - token_log_probs) - (ref_token_log_probs - token_log_probs) - 1
    
    # Create mask for valid tokens
    length_mask = mx.arange(inputs.shape[1] - 1)[None, :] < (lengths[:, None] - 1)
    
    # Compute policy ratio
    policy_ratio = mx.exp(token_log_probs - mx.stop_gradient(token_log_probs))
    
    # Compute per-token loss following GRPO formula
    per_token_loss = -(policy_ratio * advantages.reshape(-1, 1) - beta * kl_div)
    
    # Average over tokens and sequences
    sequence_sums = (per_token_loss * length_mask).sum(axis=1)
    sequence_lengths = length_mask.sum(axis=1)
    loss = (sequence_sums / sequence_lengths).mean()
    
    # Calculate mean KL divergence for metrics
    mean_kl = ((kl_div * length_mask).sum(axis=1) / length_mask.sum(axis=1)).mean()
    
    # Collect reward metrics
    reward_metrics = {}
    for i, reward_func in enumerate(reward_funcs):
        func_name = reward_func.__name__
        func_rewards = mx.array(reward_func(
            prompts=expanded_prompts,
            completions=all_completion_texts,
            answer=expanded_answers
        ))
        reward_metrics[f'{func_name}_mean'] = mx.mean(func_rewards)
        reward_metrics[f'{func_name}_std'] = mx.std(func_rewards)
    
    # Clean up
    del all_completions
    mx.metal.clear_cache()
    
    metrics = {
        'total_rewards_mean': mx.mean(rewards),
        'total_rewards_std': mx.std(rewards),
        'grouped_rewards_mean': mx.mean(rewards_reshaped),
        'grouped_rewards_std': mx.std(rewards_reshaped),
        'kl': mean_kl,
        **reward_metrics
    }
    
    return loss, sequence_lengths.sum(), metrics


def iterate_grpo_batches(dataset, tokenizer, batch_size, max_seq_length, train=False):
    """Memory-optimized version of iterate_grpo_batches"""
    if not dataset or not isinstance(dataset[0], tuple) or len(dataset[0]) != 4:
        raise ValueError("Dataset must be list of (prompt_tokens, answer_tokens, prompt_str, answer_str) tuples")

    # Sort by length but use generator to avoid keeping full sorted list in memory
    def length_key(i):
        return len(dataset[i][0]) + len(dataset[i][1])
    
    idx = sorted(range(len(dataset)), key=length_key)
    
    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size} "
            f"examples but only has {len(dataset)}."
        )

    step = mx.distributed.init().size()
    if batch_size % step != 0:
        raise ValueError("The batch size must be divisible by the number of workers")

    # Use generator for batch indices
    def batch_index_generator():
        for i in range(0, len(idx) - batch_size + 1, batch_size):
            yield idx[i : i + batch_size : step]

    while True:
        indices = (
            np.random.permutation(list(batch_index_generator())) if train 
            else batch_index_generator()
        )
        
        for batch_idx in indices:
            current_batch = [dataset[j] for j in batch_idx]
            
            prompts_tokens = [item[0] for item in current_batch]
            answers_tokens = [item[1] for item in current_batch]
            prompts_text = [item[2] for item in current_batch]
            answers_text = [item[3] for item in current_batch]

            if any(len(p) > max_seq_length for p in prompts_tokens):
                print(
                    f"[WARNING] Some prompts are longer than {max_seq_length} tokens. "
                    "Long prompts will be truncated."
                )

            yield prompts_tokens, answers_tokens, prompts_text, answers_text

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
    """
    Evaluate model using GRPO loss.
    Returns:
        tuple: (average loss, number of tokens, average metrics)
    """
    all_losses = 0
    ntokens = 0
    all_metrics = None  # Initialize metrics dictionary
    
    # Create iterator for batches
    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)
    
    # Iterate through batches
    for _, batch in zip(
        index_iterator,
        iterate_batches(
            dataset=dataset,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        ),
    ):
        # Calculate loss for current batch
        losses, toks, metrics = loss(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            reward_funcs=reward_funcs,
            beta=beta,
            group_size=group_size,
            epsilon=epsilon,
            ref_model=ref_model
        )
        
        # Accumulate losses and tokens
        all_losses += losses * toks
        ntokens += toks
        
        # Accumulate metrics
        if all_metrics is None:
            all_metrics = {k: v * toks for k, v in metrics.items()}
        else:
            for k, v in metrics.items():
                all_metrics[k] += v * toks
        
        # Evaluate accumulated values
        mx.eval(all_losses, ntokens)
    
    # Aggregate across distributed workers
    all_losses = mx.distributed.all_sum(all_losses, stream=mx.cpu)
    ntokens = mx.distributed.all_sum(ntokens, stream=mx.cpu)
    all_metrics = {k: mx.distributed.all_sum(v) for k, v in all_metrics.items()}
    
    # Calculate averages
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
        (loss, toks, metrics), grad = loss_value_and_grad(
            model, 
            tokenizer=tokenizer,
            batch=batch,
            reward_funcs=reward_funcs,
            beta=args.beta,
            group_size=args.group_size,
            epsilon=args.epsilon,
            ref_model=ref_model,
            max_tokens=args.max_seq_length,
        )

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