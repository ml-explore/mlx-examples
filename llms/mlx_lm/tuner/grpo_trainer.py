# Copyright © 2024 Apple Inc.

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from ..models import cache
from ..utils import generation_stream
from .grpo_reward_functions import (
    RewardFunctions,
    r1_accuracy_reward_func,
    r1_count_xml,
    r1_extract_xml_answer,
    r1_int_reward_func,
    r1_soft_format_reward_func,
    r1_strict_format_reward_func,
)
from .trainer import TrainingArgs, TrainingCallback, average_gradients, grad_checkpoint


@dataclass
class GRPOTrainingArgs(TrainingArgs):
    group_size: int = field(
        default=4,
        metadata={"help": "Number of responses per prompt."},
    )
    beta: float = field(default=0.1, metadata={"help": "KL penalty coefficient."})
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
        },
    )
    temperature: float = field(
        default=1.0,
        metadata={
            "help": "Temperature for sampling. The higher the temperature, the more random the completions."
        },
    )
    reward_weights: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": "Weights for each reward function. Must match the number of reward functions. If `None`, all rewards are weighted equally with weight `1.0`."
        },
    )


def generate_step(
    prompt: mx.array,
    model: nn.Module,
    *,
    max_tokens: int = 256,
    sampler: Optional[Callable[mx.array, mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    max_kv_size: Optional[int] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 512,
    prompt_progress_callback: Optional[Callable[int, int]] = None,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        max_tokens (int): The maximum number of tokens. Use``-1`` for an infinite
          generator. Default: ``256``.
        sampler (Callable[mx.array, mx.array], optional): A sampler for sampling a
          token from a vector of log probabilities. Default: ``None``.
        logits_processors (List[Callable[[mx.array, mx.array], mx.array]], optional):
          A list of functions that take tokens and logits and return the processed
          logits. Default: ``None``.
        max_kv_size (int, optional): Maximum size of the key-value cache. Old
          entries (except the first 4 tokens) will be overwritten.
        prompt_cache (List[Any], optional): A pre-computed prompt cache. Note, if
          provided, the cache will be updated in place.
        prefill_step_size (int): Step size for processing the prompt.
        kv_bits (int, optional): Number of bits to use for KV cache quantization.
          None implies no cache quantization. Default: ``None``.
        kv_group_size (int): Group size for KV cache quantization. Default: ``64``.
        quantized_kv_start (int): Step to begin using a quantized KV cache.
           when ``kv_bits`` is non-None. Default: ``0``.
        prompt_prorgress_callback (Callable[int, int]): A call-back which takes the
           prompt tokens processed so far and the total number of prompt tokens.

    Yields:
        Tuple[mx.array, mx.array]: One token and a vector of log probabilities.
    """

    y = prompt
    tokens = None

    # Create the KV cache for generation
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(
            model,
            max_kv_size=max_kv_size,
        )
    elif len(prompt_cache) != len(model.layers):
        raise ValueError("Wrong number of layers in the prompt cache.")

    prompt_progress_callback = prompt_progress_callback or (lambda *_: None)

    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

    def _step(y):
        with mx.stream(generation_stream):
            logits = model(y[None], cache=prompt_cache)
            logits = logits[:, -1, :]

            if logits_processors:
                nonlocal tokens
                tokens = mx.concat([tokens, y]) if tokens is not None else y

                for processor in logits_processors:
                    logits = processor(tokens, logits)

            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            y = sampler(logprobs)
            return mx.stop_gradient(y), mx.stop_gradient(logprobs.squeeze(0))

    with mx.stream(generation_stream):
        total_prompt_tokens = y.size
        prompt_processed_tokens = 0
        while y.size > prefill_step_size:
            model(y[:prefill_step_size][None], cache=prompt_cache)
            mx.eval([c.state for c in prompt_cache])
            prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
            prompt_processed_tokens += prefill_step_size
            y = y[prefill_step_size:]
            mx.metal.clear_cache()

        y, logprobs = _step(y)

    mx.eval(y, logprobs)
    n = 0
    while True:
        if n != max_tokens:
            next_y, next_logprobs = _step(y)
            mx.eval(next_y, next_logprobs)
        if n == 0:
            mx.eval(y)
            prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
        if n == max_tokens:
            break
        yield y.item(), logprobs
        if n % 256 == 0:
            mx.metal.clear_cache()
        y, logprobs = next_y, next_logprobs
        n += 1


def generate_grpo(
    model: nn.Module,
    prompts,
    max_tokens,
    tokenizer,
    group_size,
    end_token: str = "</answer>",
    temperature: float = 0.8,
    batch_size: int = 1,
):
    try:
        import time

        start_time = time.time()

        if len(prompts.shape) == 1:
            prompts = prompts[None, :]
        if prompts.shape[1] == 0:
            return None

        total_samples = prompts.shape[0] * group_size
        expanded_prompts = mx.repeat(prompts, group_size, axis=0)
        end_sequence = mx.array(tokenizer.encode(end_token))
        results = []
        mx.eval(expanded_prompts, results)

        print(f"Setup time: {time.time() - start_time:.2f}s")
        print(f"Generating {total_samples} samples with max_tokens={max_tokens}")

        total_tokens_generated = 0
        generation_start_time = time.time()

        # Process in batches
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_time = time.time()
            print(
                f"Starting batch {batch_start//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}: samples {batch_start}-{batch_end-1}"
            )

            # Custom sampler function that handles temperature
            def temp_sampler(logits):
                return mx.random.categorical(logits / temperature)

            # Batched processing
            for idx in range(batch_start, batch_end):
                sample_start_time = time.time()
                current_tokens = []
                prompt_cache = cache.make_prompt_cache(model)
                mx.eval(current_tokens, prompt_cache)

                # The generate_step function yields one token at a time
                # We'll collect tokens until we hit max_tokens or a stopping condition
                for i, (token, _) in enumerate(
                    generate_step(
                        expanded_prompts[idx],
                        model,
                        max_tokens=max_tokens,  # This is the maximum number of steps
                        sampler=temp_sampler,
                        prompt_cache=prompt_cache,
                    )
                ):
                    print(token)

                    # Check for EOS token
                    if token == tokenizer.eos_token_id:
                        break
                    
                    current_tokens.append(token)

                    # Check for end token
                    if len(current_tokens) >= len(end_sequence) and mx.array_equal(
                        mx.array(current_tokens[-len(end_sequence) :]), end_sequence
                    ):
                        break

                    # Check if we've reached the maximum number of tokens
                    if i >= max_tokens - 1:
                        break

                if current_tokens:
                    results.append(mx.array(current_tokens))
                    total_tokens_generated += len(current_tokens)

                sample_time = time.time() - sample_start_time
                tokens_per_second = (
                    len(current_tokens) / sample_time if sample_time > 0 else 0
                )
                print(
                    f"  Sample {idx}: Generated {len(current_tokens)} tokens in {sample_time:.2f}s ({tokens_per_second:.2f} tokens/sec)"
                )

            batch_time = time.time() - batch_time
            print(f"Batch completed in {batch_time:.2f}s")
            mx.metal.clear_cache()

        generation_time = time.time() - generation_start_time
        avg_tokens_per_second = (
            total_tokens_generated / generation_time if generation_time > 0 else 0
        )

        print(
            f"Generation complete: {total_tokens_generated} tokens in {generation_time:.2f}s"
        )
        print(f"Average generation speed: {avg_tokens_per_second:.2f} tokens/sec")

        mx.eval(results)
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
            log_probs, seq_targets.reshape(seq_len, 1), axis=-1
        ).squeeze(-1)
        per_token_logps.append(token_log_probs)
    mx.eval(logits)
    return per_token_logps


def grpo_loss(
    model,
    ref_model,
    tokenizer,
    batch,
    reward_funcs: Optional[List[RewardFunctions]] = None,
    beta: float = 0.1,
    group_size: int = 4,
    epsilon: float = 1e-4,
    max_tokens: int = 64,
    temperature: float = 0.8,
    reward_weights: Optional[List[float]] = None,
    batch_size: int = 1,
):
    prompt_tokens, _, prompt_text, answer_text = batch
    total_samples = len(prompt_tokens)

    all_completions = []
    all_completion_texts = []
    batch_indices = []  # Keep track of which batch each completion belongs to

    # Store original training state
    was_training = model.training
    print(f"Was model in training mode: {was_training}")

    # Set model to eval mode for generation
    model.eval()

    # Process in smaller batches
    for i in range(0, total_samples, batch_size):
        # Get actual batch size for this iteration (might be smaller for the last batch)
        current_batch_size = min(batch_size, total_samples - i)
        batch_prompts = prompt_tokens[i : i + current_batch_size]

        # Pad sequences to the same length
        max_prompt_len = max(len(p) for p in batch_prompts)
        padded_prompts = []

        for prompt in batch_prompts:
            padding = [tokenizer.pad_token_id] * (max_prompt_len - len(prompt))
            padded_prompts.append(prompt + padding)

        # Convert to tensor
        prompt_tensor = mx.array(padded_prompts)

        try:
            completions = generate_grpo(
                model,
                prompt_tensor,
                max_tokens,
                tokenizer,
                group_size,
                temperature=temperature,
                batch_size=current_batch_size,
            )

            if completions is not None:
                for j, completion_ids in enumerate(completions):
                    # Calculate which prompt this completion belongs to
                    prompt_idx = i + (j // group_size)
                    
                    if prompt_idx < total_samples:  # Make sure we don't go out of bounds
                        batch_indices.append(prompt_idx)
                        completion_text = tokenizer.decode(completion_ids.tolist())
                        all_completions.append(completion_ids)
                        all_completion_texts.append(completion_text)
                        mx.eval(completion_ids)
        except Exception as e:
            print(f"Generation error: {e}")
            continue

    # Restore original training state if we're not in validation mode
    if was_training:
        model.train()
    mx.metal.clear_cache()

    # If we didn't generate any completions, return early
    if not all_completions:
        raise ValueError(
            "No completions were generated. Please check your model and inputs."
        )

    # Create expanded prompts and answers based on actual generated completions
    expanded_answers = []
    expanded_prompts = []

    # Group completions by their original prompt
    unique_prompt_indices = sorted(set(batch_indices))
    grouped_completions = {idx: [] for idx in unique_prompt_indices}

    for i, completion_idx in enumerate(batch_indices):
        grouped_completions[completion_idx].append(i)

    # Rebuild completions in the correct order
    ordered_completions = []
    ordered_completion_texts = []
    ordered_batch_indices = []

    for prompt_idx in unique_prompt_indices:
        completion_indices = grouped_completions[prompt_idx]
        for idx in completion_indices:
            ordered_completions.append(all_completions[idx])
            ordered_completion_texts.append(all_completion_texts[idx])
            ordered_batch_indices.append(prompt_idx)

            # Add corresponding prompt and answer
            expanded_prompts.append(prompt_text[prompt_idx])
            expanded_answers.append(answer_text[prompt_idx])

    all_completions = ordered_completions
    all_completion_texts = ordered_completion_texts
    batch_indices = ordered_batch_indices

    # Continue with the rest of the function
    max_length = max(ids.shape[0] for ids in all_completions)
    padded_completions = []
    attention_masks = []

    for completion_ids in all_completions:
        padding_length = max_length - completion_ids.shape[0]
        if padding_length > 0:
            padding = mx.zeros((padding_length,), dtype=completion_ids.dtype)
            padded_ids = mx.concatenate([completion_ids, padding])
            mask = mx.concatenate(
                [mx.ones_like(completion_ids), mx.zeros_like(padding)]
            )
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
        func_rewards = mx.array(
            reward_func(
                prompts=expanded_prompts,
                completions=all_completion_texts,
                answer=expanded_answers,
            )
        )
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

    # Get number of unique prompts
    num_unique_prompts = len(unique_prompt_indices)

    # Reshape rewards based on actual groups
    rewards_by_prompt = [[] for _ in range(num_unique_prompts)]
    for i, prompt_idx in enumerate(batch_indices):
        prompt_position = unique_prompt_indices.index(prompt_idx)
        rewards_by_prompt[prompt_position].append(rewards[i])

    # Calculate advantages within each group
    advantages = mx.zeros_like(rewards)
    for i, prompt_rewards in enumerate(rewards_by_prompt):
        if len(prompt_rewards) > 1:  # Only normalize if we have multiple samples
            prompt_rewards = mx.array(prompt_rewards)
            mean_reward = mx.mean(prompt_rewards)
            std_reward = mx.std(prompt_rewards)

            # Find indices for this prompt
            indices = [
                j
                for j, idx in enumerate(batch_indices)
                if idx == unique_prompt_indices[i]
            ]
            for j, idx in enumerate(indices):
                advantages[idx] = (prompt_rewards[j] - mean_reward) / (
                    std_reward + epsilon
                )
        else:
            # If only one sample, advantage is 0
            idx = batch_indices.index(unique_prompt_indices[i])
            advantages[idx] = 0.0

    # Compute KL divergence using Schulman's approximator
    kl_div = (
        mx.exp(ref_token_log_probs - token_log_probs)
        - (ref_token_log_probs - token_log_probs)
        - 1
    )

    # Create mask for valid tokens
    length_mask = mx.arange(inputs.shape[1] - 1)[None, :] < (lengths[:, None] - 1)

    # Compute policy ratio
    policy_ratio = mx.exp(
        mx.array(token_log_probs - mx.stop_gradient(ref_token_log_probs))
    )

    # Compute per-token loss
    per_token_loss = -(
        (policy_ratio * advantages.reshape(-1, 1) - beta * kl_div) * length_mask
    )

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
        func_rewards = mx.array(
            reward_func(
                prompts=expanded_prompts,
                completions=all_completion_texts,
                answer=expanded_answers,
            )
        )
        reward_metrics[f"{func_name}_mean"] = mx.mean(func_rewards)
        reward_metrics[f"{func_name}_std"] = mx.std(func_rewards)

    grouped_rewards_mean = mx.array(
        [mx.mean(mx.array(rewards)) for rewards in rewards_by_prompt]
    )
    grouped_rewards_std = mx.array(
        [
            mx.std(mx.array(rewards)) if len(rewards) > 1 else mx.zeros(1)
            for rewards in rewards_by_prompt
        ]
    )

    metrics = {
        "total_rewards_mean": mx.mean(rewards),
        "total_rewards_std": mx.std(rewards),
        "grouped_rewards_mean": mx.mean(grouped_rewards_mean),
        "grouped_rewards_std": mx.mean(grouped_rewards_std),
        "kl": mean_kl,
        **reward_metrics,
    }

    if is_validation and all_completion_texts:
        print("\n=== Validation Sample Details ===")

        # Print the input context (prompt)
        last_prompt_idx = batch_indices[-1] if batch_indices else 0

        if last_prompt_idx < len(prompt_text):
            print(f"\n📋 Raw Prompt:\n{prompt_text[last_prompt_idx]}")
            print("\n" + "=" * 10 + "\n")

            # Get the actual tokenized prompt that was fed to the model
            if last_prompt_idx < len(prompt_tokens):
                actual_prompt = tokenizer.decode(prompt_tokens[last_prompt_idx])
                print(f"\n🔄 Model Input:\n{actual_prompt}")
                print("\n" + "=" * 10 + "\n")

        print(f"\n📝 Generation:\n{all_completion_texts[-1]}")
        print("\n" + "=" * 10 + "\n")

        # Make sure we have a valid index for answer_text
        if last_prompt_idx < len(answer_text):
            print(f"\n✅ Answer:\n{answer_text[last_prompt_idx]}")
            print("\n" + "=" * 10 + "\n")

        # Only try to extract if r1_extract_xml_answer is defined
        if "r1_extract_xml_answer" in globals():
            print(
                f"\n🔍 Extracted Answer:\n{r1_extract_xml_answer(all_completion_texts[-1])}"
            )
        print("\n" + "=" * 35 + "\n")

    mx.metal.clear_cache()

    return loss, sequence_lengths.sum(), metrics


def iterate_grpo_batches(dataset, batch_size, max_seq_length, train=False):
    if not dataset or not isinstance(dataset[0], tuple) or len(dataset[0]) != 4:
        raise ValueError(
            "Dataset must be list of (prompt_tokens, answer_tokens, prompt_str, answer_str) tuples"
        )

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
            np.random.permutation(list(batch_index_generator()))
            if train
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
    reward_funcs: Optional[List[RewardFunctions]] = [
        r1_accuracy_reward_func,
        r1_int_reward_func,
        r1_strict_format_reward_func,
        r1_soft_format_reward_func,
        r1_count_xml,
    ],
    loss_fn: callable = grpo_loss,
    iterate_batches: callable = iterate_grpo_batches,
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
            max_tokens=max_tokens
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
        r1_count_xml,
    ],
    args: GRPOTrainingArgs = GRPOTrainingArgs(),
    loss_fn: callable = grpo_loss,
    iterate_batches: callable = iterate_grpo_batches,
    training_callback: TrainingCallback = None,
):
    print(
        f"Starting GRPO training with {len(reward_funcs)} reward functions..., iters: {args.iters}"
    )
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
            temperature=args.temperature,
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
        "total_rewards_mean": 0,
        "total_rewards_std": 0,
        "grouped_rewards_mean": 0,
        "grouped_rewards_std": 0,
        "kl": 0,
    }
    for reward_func in reward_funcs:
        func_name = reward_func.__name__
        accumulated_metrics[f"{func_name}_mean"] = 0
        accumulated_metrics[f"{func_name}_std"] = 0

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
                    f"Val loss {val_loss:.3f}, "
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
                    f"Iter {it}: {val_metrics_str}, " f"Val took {val_time:.3f}s",
                    flush=True,
                )

            if training_callback is not None:
                training_callback.on_val_loss_report(
                    {
                        "iteration": it,
                        "val_loss": val_loss,
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                        "val_time": val_time,
                    }
                )

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
            avg_metrics = {
                k: v / (steps * world_size) for k, v in accumulated_metrics.items()
            }
            n_tokens = mx.distributed.all_sum(n_tokens, stream=mx.cpu).item()
            learning_rate = optimizer.learning_rate.item()
            it_sec = args.steps_per_report / (stop - start)
            tokens_sec = float(n_tokens) / (stop - start)
            trained_tokens += n_tokens
            peak_mem = mx.metal.get_peak_memory() / 1e9

            if rank == 0:
                train_metrics_str = (
                    f"Train loss {train_loss:.3f}, "
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
                training_callback.on_train_loss_report(
                    {
                        "iteration": it,
                        "train_loss": train_loss,
                        **{f"train_{k}": v for k, v in avg_metrics.items()},
                        "learning_rate": learning_rate,
                        "iterations_per_second": it_sec,
                        "tokens_per_second": tokens_sec,
                        "trained_tokens": trained_tokens,
                        "peak_memory": peak_mem,
                    }
                )

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
