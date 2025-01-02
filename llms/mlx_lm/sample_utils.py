# Copyright © 2023-2024 Apple Inc.

import math
from functools import partial
from typing import Callable, Dict, Optional

import mlx.core as mx


def make_sampler(
    temp: float = 0.0,
    top_p: float = 0.0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
    top_k: int = -1,
) -> Callable[mx.array, mx.array]:
    """
    Make a sampler function for use with ``generate_step``.

    Args:
        temp (float): The temperature for sampling, if 0 the argmax is used.
          Default: ``0``.
        top_p (float, optional): Nulceus sampling, higher means model considers
          more less likely words.
        min_p (float, optional): The minimum value (scaled by the top token's
          probability) that a token probability must have to be considered.
        min_tokens_to_keep (int, optional): Minimum number of tokens that cannot
          be filtered by min_p sampling.
        top_k (int, optional): The top k tokens ranked by probability to constrain
          the sampling to.

    Returns:
        Callable[mx.array, mx.array]:
            A sampler which takes log-probabilities and returns tokens.
    """
    if temp == 0:
        return lambda x: mx.argmax(x, axis=-1)
    elif top_p > 0 and top_p < 1.0:
        return lambda x: top_p_sampling(x, top_p, temp)
    elif min_p != 0.0:
        return lambda x: min_p_sampling(x, min_p, min_tokens_to_keep, temp)
    elif top_k > 0:
        return lambda x: top_k_sampling(x, top_k, temp)
    else:
        return lambda x: categorical_sampling(x, temp)


def make_logits_processors(
    logit_bias: Optional[Dict[int, float]] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
):
    """
    Make logits processors for use with ``generate_step``.

    Args:
        repetition_penalty (float, optional): The penalty factor for repeating
          tokens.
        repetition_context_size (int, optional): The number of tokens to
          consider for repetition penalty. Default: ``20``.
        logit_bias (dictionary, optional): Additive logit bias.

    Returns:
        List[Callable[[mx.array, mx.array], mx.array]]:
            A list of logits processors. Each processor in the list is a
            callable which takes an array of tokens and an array of logits
            and returns the updated logits.
    """
    logits_processors = []
    if logit_bias:
        indices = mx.array(list(logit_bias.keys()))
        values = mx.array(list(logit_bias.values()))

        def logit_bias_processor(_, logits):
            logits[..., indices] += values
            return logits

        logits_processors.append(logit_bias_processor)

    if repetition_penalty and repetition_penalty != 0.0:
        logits_processors.append(
            make_repetition_penalty(repetition_penalty, repetition_context_size)
        )
    return logits_processors


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def top_k_sampling(
    logprobs: mx.array,
    top_k: int,
    temperature=1.0,
) -> mx.array:
    """
    Sample from only the top K tokens ranked by probability.

    Args:
        logprobs: A vector of log probabilities.
        top_k (int): Top k tokens to sample from.
    """
    vocab_size = logprobs.shape[-1]
    if not isinstance(top_k, int) or not (0 < top_k < vocab_size):
        raise ValueError(
            f"`top_k` has to be an integer in the (0, {vocab_size}] interval,"
            f" but is {top_k}."
        )
    logprobs = logprobs * (1 / temperature)
    mask_idx = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., top_k:]
    masked_logprobs = mx.put_along_axis(
        logprobs, mask_idx, mx.array(-float("inf"), logprobs.dtype), axis=-1
    )
    return mx.random.categorical(masked_logprobs, axis=-1)


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def min_p_sampling(
    logprobs: mx.array,
    min_p: float,
    min_tokens_to_keep: int = 1,
    temperature=1.0,
) -> mx.array:
    """
    Apply min-p sampling to the logprobs.

    Min-p keeps all tokens that are above a minimum probability, scaled by the
    probability of the most likely token. As a result, the filter is more
    aggressive given a very high-probability token.

    Args:
        logprobs: A vector of log probabilities.
        min_p (float): Minimum token probability. Typical values are in the
            0.01-0.2 range, comparably selective as setting `top_p` in the
            0.99-0.8 range.
        min_tokens_to_keep (int, optional): Minimum number of tokens that cannot
            be filtered. Default: ``1``.
        temperature: Temperature parameter for softmax distribution reshaping.
    Returns:
        token(s) selected based on the min-p criterion.
        Shape: same as logits, but with the last dimension having size 1.
    """
    if not (0 <= min_p <= 1.0):
        raise ValueError(
            f"`min_p` has to be a float in the [0, 1] interval, but is {min_p}"
        )
    if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
        raise ValueError(
            f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}"
        )
    # reference implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L531-L605

    logprobs = logprobs * (1 / temperature)

    # Indices sorted in decreasing order
    sorted_indices = mx.argsort(-logprobs)
    sorted_logprobs = mx.take_along_axis(logprobs, sorted_indices, axis=-1)

    # Top probability
    top_logprobs = mx.expand_dims(sorted_logprobs[..., 0], axis=-1)

    # Calculate the min_p threshold
    scaled_min_p = top_logprobs + math.log(min_p)

    # Mask tokens that have a probability less than the scaled min_p
    tokens_to_remove = sorted_logprobs < scaled_min_p
    tokens_to_remove[..., :min_tokens_to_keep] = False

    # Create pool of tokens with probability less than scaled min_p
    selected_logprobs = mx.where(tokens_to_remove, -float("inf"), sorted_logprobs)

    # Return sampled token(s)
    sampled_indices = mx.random.categorical(selected_logprobs)
    tokens = mx.take_along_axis(
        sorted_indices, mx.expand_dims(sampled_indices, axis=-1), axis=-1
    )
    return tokens.squeeze(-1)


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def top_p_sampling(
    logits: mx.array, top_p: float, temperature: float, axis: int = -1
) -> mx.array:
    """
    Apply top-p (nucleus) sampling to logits.

    Args:
        logits: The logits from the model's output.
        top_p: The cumulative probability threshold for top-p filtering.
        temperature: Temperature parameter for softmax distribution reshaping.
        axis: The axis along which to apply top-p sampling.
    Returns:
        token(s) selected based on the top-p criterion.
    """
    # referenced implementation from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L449-L460
    # Apply temperature and compute softmax
    probs = mx.softmax(logits / temperature, axis=axis)

    # Sort probs in descending order
    sorted_indices = mx.argsort(-probs, axis=axis)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=axis)

    # Compute cumulative probabilities
    cumulative_probs = mx.cumsum(sorted_probs, axis=axis)

    # Create a mask for probs above the threshold
    mask = cumulative_probs <= top_p

    # Apply the mask to the sorted probabilities
    masked_probs = sorted_probs * mask

    # Sample from the normalized probabilities
    sampled_indices = mx.random.categorical(mx.log(masked_probs), axis=axis)

    # Gather the original token indices
    tokens = mx.take_along_axis(
        sorted_indices, mx.expand_dims(sampled_indices, axis=axis), axis=axis
    )

    return tokens.squeeze(axis)


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def categorical_sampling(logits, temp):
    return mx.random.categorical(logits * (1 / temp))


def make_repetition_penalty(penalty: float, context_size: int = 20):
    """
    Make repetition penalty processor.

    Paper: https://arxiv.org/abs/1909.05858

    Args:
        penalty (float): The repetition penalty factor to be applied.
        context_size (int): The number of previous tokens to use.
            Default: ``20``.

    Returns:
        Callable[[mx.array, List[int]], mx.array]:
            The repetition penalty processor.
    """
    if penalty < 0 or not isinstance(penalty, (int, float)):
        raise ValueError(f"penalty must be a non-negative float, got {penalty}")

    def repetition_penalty_processor(tokens, logits):
        if tokens.shape[-1] > 0:
            tokens = tokens[..., -context_size:]
            selected_logits = mx.take_along_axis(logits, tokens, axis=-1)
            selected_logits = mx.where(
                selected_logits < 0,
                selected_logits * penalty,
                selected_logits / penalty,
            )
            logits[mx.arange(tokens.shape[0])[:, None], tokens] = selected_logits
        return logits

    return repetition_penalty_processor
