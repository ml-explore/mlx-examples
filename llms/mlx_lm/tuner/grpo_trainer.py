# Copyright © 2024 Apple Inc.

from typing import List, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import time
import re

from mlx.utils import tree_flatten
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .trainer import grad_checkpoint, TrainingArgs, TrainingCallback, average_gradients


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
    max_completion_length: int = field(
        default=512, metadata={"help": "Number of Generations."}
    )
    reference_model_path: str = field(
        default=None,
        metadata={
            "help": "Path to reference model weights. If None, uses the same model."
        }
    )
    temperature: float = field(
        default=1.0,
        metadata={
            "help": "Temperature for sampling. The higher the temperature, the more random the completions."
        }
    )
    reward_weights: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": "Weights for each reward function. Must match the number of reward functions. If `None`, all rewards are weighted equally with weight `1.0`."
        }
    )


RewardFunctions = Callable[[List[str], List[str], List[str]], List[float]]


def r1_extract_xml_answer(text: str) -> str:
    try:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    except:
        print("r1_extract_xml_answer returned empty string")
        return ""


def r1_int_reward_func(prompts: list, completions: list, answer: list, **kwargs) -> list[float]:
    if not completions:
        return [0.0] * len(prompts)
    extracted_responses = [r1_extract_xml_answer(r) for r in completions]
    return [0.5 if r and r.isdigit() else 0.0 for r in extracted_responses]

def r1_accuracy_reward_func(prompts: list, completions: list, answer: list, **kwargs) -> list[float]:
    if not completions or not answer:
        return [0.0] * len(prompts)
    extracted_responses = [r1_extract_xml_answer(r) for r in completions]
    return [2.0 if r and a and r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def r1_soft_format_reward_func(prompts: list, completions: list, answer: list, **kwargs) -> list[float]:
    if not completions:
        return [0.0] * len(prompts)
    has_think = r"<think>.*</think>"
    has_answer = r"<answer>.*</answer>"
    matches = [(bool(re.search(has_think, r)) and bool(re.search(has_answer, r))) if r else False for r in completions]
    return [0.5 if match else 0.0 for match in matches]

def r1_strict_format_reward_func(prompts: list, completions: list, answer: list, **kwargs) -> list[float]:
    if not completions:
        return [0.0] * len(prompts)
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    matches = [bool(re.search(pattern, r)) if r else False for r in completions]
    return [0.5 if match else 0.0 for match in matches]


def r1_count_xml(prompts: list, completions: list, answer: list, **kwargs) -> list[float]:
    if not completions:
        return [0.0] * len(prompts)
    scores = []
    for text in completions:
        if not text:
            scores.append(0.0)
            continue
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.125
        if text.count("\n</think>\n") == 1:
            count += 0.125
        if text.count("\n<answer>\n") == 1:
            count += 0.125
        if text.count("\n</answer>\n") == 1:
            count += 0.125
        end_text = text.split("\n</answer>\n")[-1]
        count -= len(end_text) * 0.001 if len(end_text) > 0 else 0
        scores.append(max(0.0, count))
    return scores


def generate_grpo(model: nn.Module, prompts, max_tokens, tokenizer, temperature, group_size):
    if len(prompts.shape) == 1:
        prompts = prompts[None, :]
    if prompts.shape[1] == 0:
        return None

    start_time = time.perf_counter()
    tokens_generated = 0
    batch_size = prompts.shape[0] * group_size

    # Repeat each prompt group_size times
    expanded_prompts = mx.repeat(prompts, group_size, axis=0)
    
    end_sequence = mx.array(tokenizer.encode("</answer>"))
    end_len = len(end_sequence)
    initial_length = prompts.shape[1]
    
    # Initialize output tensor for all sequences
    output = mx.zeros((batch_size, initial_length + max_tokens), dtype=mx.int32)
    output = mx.concatenate([expanded_prompts, mx.zeros((batch_size, max_tokens), dtype=mx.int32)], axis=1)
    current_lengths = mx.array([initial_length] * batch_size)
    
    temp_factor = 1/temperature if temperature > 0 else float('inf')
    
    try:
        not_finished = mx.ones((batch_size,), dtype=mx.bool_)
        
        for _ in range(max_tokens):
            # Check if all sequences are finished
            if not mx.sum(not_finished).item():
                break
                
            # Get model outputs for all sequences
            max_len = mx.max(current_lengths).item()
            batch_inputs = output[:, :max_len]
            logits = model(batch_inputs)[:, -1]
            
            # Apply mask to logits
            logits = logits * mx.expand_dims(not_finished, -1)
            
            # Sample next tokens
            logits *= temp_factor
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            next_tokens = mx.random.categorical(logprobs)
            
            # Update outputs for active sequences
            for idx in range(batch_size):
                if not_finished[idx].item():
                    curr_len = current_lengths[idx].item()
                    token_value = next_tokens[idx].item()
                    
                    # Create new arrays with updates
                    output = mx.array(output.tolist())  # Make a copy
                    output[idx, curr_len] = token_value
                    current_lengths = mx.array([
                        l + 1 if i == idx else l 
                        for i, l in enumerate(current_lengths.tolist())
                    ])
                    tokens_generated += 1
                    
                    # Check end conditions
                    if token_value == tokenizer.eos_token_id:
                        not_finished = mx.array([
                            False if i == idx else nf 
                            for i, nf in enumerate(not_finished.tolist())
                        ])
                    elif curr_len >= end_len:
                        last_tokens = output[idx, curr_len-end_len+1:curr_len+1]
                        if mx.array_equal(last_tokens, end_sequence):
                            not_finished = mx.array([
                                False if i == idx else nf 
                                for i, nf in enumerate(not_finished.tolist())
                            ])
            
            if _ % 32 == 0:
                mx.eval(output, current_lengths, not_finished)

        end_time = time.perf_counter()
        generation_time = end_time - start_time
        tokens_per_second = tokens_generated / generation_time
        print(f"Generated {tokens_generated} tokens in {generation_time:.2f}s ({tokens_per_second:.2f} tokens/s)")

        # Return only the valid part of each sequence
        results = [output[i, :current_lengths[i].item()] for i in range(batch_size)]
        return results

    except Exception as e:
        print(f"Generation error: {str(e)}")
        return None


def get_per_token_logps(model: nn.Module, inputs, lengths):
    logits = model(inputs).astype(mx.float16)
    logits = logits[:, :-1, :]
    targets = inputs[:, 1:]
    per_token_logps = []
    for i in range(logits.shape[0]):
        seq_len = int(lengths[i]) - 1
        seq_logits = logits[i, :seq_len]
        seq_targets = targets[i, :seq_len]
        log_probs = nn.log_softmax(seq_logits, axis=-1)
        token_log_probs = mx.take_along_axis(
            log_probs,
            seq_targets.reshape(seq_len, 1),
            axis=-1
        ).squeeze(-1)
        per_token_logps.append(token_log_probs)
        mx.eval(logits)
    return per_token_logps


def grpo_loss(
    model,
    ref_model,
    tokenizer,
    batch,
    reward_funcs=None,
    beta=0.1,
    group_size=4,
    epsilon=1e-4,
    max_tokens=64,
    temperature=1.0,
    reward_weights=None,
    is_validation=False
):
    prompt_tokens, _, prompt_text, answer_text = batch
    batch_size = len(prompt_tokens)

    all_completions = []
    all_completion_texts = []
    
    for i in range(0, batch_size, batch_size):
        batch_prompts = prompt_tokens[i:i+batch_size]
        prompt_tensor = mx.stack([mx.array(p) for p in batch_prompts])
        
        try:
            completions = generate_grpo(
                model, 
                prompt_tensor, 
                max_tokens, 
                tokenizer, 
                temperature,
                group_size
            )
            
            if completions is not None:
                for completion_ids in completions:
                    completion_text = tokenizer.decode(completion_ids.tolist())
                    all_completions.append(completion_ids)
                    all_completion_texts.append(completion_text)
                    mx.eval(completion_ids)
        except Exception as e:
            print(f"Generation error: {e}")
            continue

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
    
    inputs = mx.stack(padded_completions)
    attention_mask = mx.stack(attention_masks)
    lengths = attention_mask.sum(axis=1)
    
    # Current policy probabilities
    token_log_probs = get_per_token_logps(model, inputs, lengths)

    mx.eval(token_log_probs)

    if ref_model is None:
        ref_token_log_probs = token_log_probs
    else:
        ref_token_log_probs = get_per_token_logps(ref_model, inputs, lengths)
        mx.eval(ref_token_log_probs)
        
    max_len = max(x.shape[0] for x in token_log_probs)
    padded_log_probs = []
    padded_ref_log_probs = []

    for i in range(len(token_log_probs)):
        seq_len = token_log_probs[i].shape[0]
        padding = mx.zeros((max_len - seq_len,))
        
        padded_log_probs.append(mx.concatenate([token_log_probs[i], padding]))
        padded_ref_log_probs.append(mx.concatenate([ref_token_log_probs[i], padding]))
    
    token_log_probs = mx.stack(padded_log_probs)
    ref_token_log_probs = mx.stack(padded_ref_log_probs)
    
    # Create array to store rewards from each function
    all_func_rewards = []

    # Collect rewards from each function separately
    for reward_func in reward_funcs:
        func_rewards = mx.array(reward_func(
            prompts=expanded_prompts,
            completions=all_completion_texts,
            answer=expanded_answers
        ))
        all_func_rewards.append(func_rewards)

    # Stack rewards to shape (num_samples, num_funcs)
    rewards = mx.stack(all_func_rewards, axis=1)

    # Apply weights and sum
    if reward_weights is not None:
        if len(reward_weights) != len(reward_funcs):
            raise ValueError(
                f"Number of reward weights ({len(reward_weights)}) must match number of reward "
                f"functions ({len(reward_funcs)})"
            )
        reward_weights = mx.array(reward_weights, dtype=mx.float32)
    else:
        reward_weights = mx.ones(len(reward_funcs), dtype=mx.float32)
        rewards = (rewards * mx.expand_dims(reward_weights, 0)).sum(axis=1)

    # Reshape rewards and compute advantages
    rewards_reshaped = rewards.reshape(batch_size, group_size)
    mean_rewards = mx.broadcast_to(mx.mean(rewards_reshaped, axis=1)[:, None], (rewards_reshaped.shape[0], group_size)).reshape(-1)
    std_rewards = mx.broadcast_to(mx.std(rewards_reshaped, axis=1)[:, None], (rewards_reshaped.shape[0], group_size)).reshape(-1)
    advantages = (rewards - mean_rewards) / (std_rewards + epsilon)
    
    # Compute KL divergence using Schulman's approximator
    kl_div = mx.exp(ref_token_log_probs - token_log_probs) - (ref_token_log_probs - token_log_probs) - 1
    
    # Create mask for valid tokens
    length_mask = mx.arange(inputs.shape[1] - 1)[None, :] < (lengths[:, None] - 1)
    
    # Compute policy ratio
    policy_ratio = mx.exp(mx.array(token_log_probs - mx.stop_gradient(ref_token_log_probs)))
    
    # Compute per-token loss
    per_token_loss = -((policy_ratio * advantages.reshape(-1, 1) - beta * kl_div) * length_mask)
    
    # Average over tokens
    sequence_sums = per_token_loss.sum(axis=1)
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
    
    metrics = {
        'total_rewards_mean': mx.mean(rewards),
        'total_rewards_std': mx.std(rewards),
        'grouped_rewards_mean': mx.mean(rewards_reshaped),
        'grouped_rewards_std': mx.std(rewards_reshaped),
        'kl': mean_kl,
        **reward_metrics
    }

    if is_validation:
        print(f"\nValidation sample generation:\n{all_completion_texts[-1]}\n")

    mx.metal.clear_cache()
    
    return loss, sequence_lengths.sum(), metrics


def iterate_grpo_batches(dataset, batch_size, max_seq_length, train=False):
    if not dataset or not isinstance(dataset[0], tuple) or len(dataset[0]) != 4:
        raise ValueError("Dataset must be list of (prompt_tokens, answer_tokens, prompt_str, answer_str) tuples")

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
    model: nn.Module,
    ref_model: Optional[nn.Module],
    dataset,
    tokenizer,
    batch_size,
    num_batches,
    beta: float,
    epsilon: float,
    group_size: int,
    max_seq_length: int,
    max_tokens: int,
    temperature: float,
    reward_funcs: Optional[List[RewardFunctions]] = None,
    loss_fn: callable = grpo_loss,
    iterate_batches: callable = iterate_grpo_batches
):
    all_losses = 0
    ntokens = 0
    all_metrics = None
    
    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)
    
    for _, batch in zip(
        index_iterator,
        iterate_batches(
            dataset=dataset,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        ),
    ):
        losses, toks, metrics = loss_fn(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            reward_funcs=reward_funcs,
            beta=beta,
            group_size=group_size,
            epsilon=epsilon,
            ref_model=ref_model,
            temperature=temperature,
            max_tokens=max_tokens,
            is_validation=True
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
    model: nn.Module,
    ref_model: Optional[nn.Module],
    tokenizer,
    optimizer,
    train_dataset,
    val_dataset,
    reward_funcs: Optional[List[RewardFunctions]] = [
        r1_accuracy_reward_func,
        r1_int_reward_func,
        r1_strict_format_reward_func,
        r1_soft_format_reward_func,
        r1_count_xml
    ],
    args: GRPOTrainingArgs = GRPOTrainingArgs(),
    loss_fn: callable = grpo_loss,
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
        (loss, toks, metrics), grad = loss_value_and_grad(
            model, 
            tokenizer=tokenizer,
            batch=batch,
            reward_funcs=reward_funcs,
            beta=args.beta,
            group_size=args.group_size,
            epsilon=args.epsilon,
            ref_model=ref_model,
            max_tokens=args.max_completion_length,
            temperature=args.temperature
        )

        grad = average_gradients(grad)

        optimizer.update(model, grad)

        return loss, toks, metrics
    
    loss_value_and_grad = nn.value_and_grad(model, loss_fn)

    losses = 0
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    accumulated_metrics = {
        'total_rewards_mean': 0,
        'total_rewards_std': 0,
        'grouped_rewards_mean': 0,
        'grouped_rewards_std': 0,
        'kl': 0
    }
    for reward_func in reward_funcs:
        func_name = reward_func.__name__
        accumulated_metrics[f'{func_name}_mean'] = 0
        accumulated_metrics[f'{func_name}_std'] = 0

    start = time.perf_counter()
    for it, batch in zip(
        range(1, args.iters + 1),
        iterate_batches(
            dataset=train_dataset,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            train=True,
        ),
    ):
        if it == 1 or it % args.steps_per_eval == 0 or it == args.iters:
            stop = time.perf_counter()
            val_loss, val_ntokens, val_metrics = evaluate_grpo(
                model=model,
                dataset=val_dataset,
                loss_fn=loss_fn,
                ref_model=ref_model,
                reward_funcs=reward_funcs,
                tokenizer=tokenizer,
                group_size=args.group_size,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                max_tokens=args.max_completion_length,
                beta=args.beta,
                epsilon=args.epsilon,
                temperature=args.temperature,
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
                
                for i, reward_func in enumerate(reward_funcs):
                    val_metrics_str += (
                        f", Val {reward_func.__name__}_mean {val_metrics[f'{reward_func.__name__}_mean']:.3f}, "
                        f"Val {reward_func.__name__}_std {val_metrics[f'{reward_func.__name__}_std']:.3f}"
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
                
                for i, reward_func in enumerate(reward_funcs):
                    func_name = reward_func.__name__
                    train_metrics_str += (
                        f", {func_name} mean {avg_metrics[f'{func_name}_mean']:.3f}, "
                        f"{func_name} std {avg_metrics[f'{func_name}_std']:.3f}"
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

    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(args.adapter_file), adapter_weights)
    print(f"Saved final weights to {args.adapter_file}.")