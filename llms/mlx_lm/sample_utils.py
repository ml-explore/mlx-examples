# Copyright © 2023-2024 Apple Inc.

from functools import partial

import mlx.core as mx


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def min_p_sampling(
    logits: mx.array,
    min_p: float,
    min_tokens_to_keep: int = 1,
    temperature=1.0,
) -> mx.array:
    """
    Apply min-p sampling to the logits.

    Min-p keeps all tokens that are above a minimum probability, scaled by the
    probability of the most likely token. As a result, the filter is more
    aggressive given a very high-probability token.

    Args:
        logits: The logits from the model's output.
        min_p (float): Minimum token probability. Typical values are in the
            0.01-0.2 range, comparably selective as setting `top_p` in the
            0.99-0.8 range.
        min_tokens_to_keep (int, optional): Minimum number of tokens that cannot
            be filtered. Default: ``1``.

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

    # Softmax probabilities
    probs = mx.softmax(logits * (1 / temperature), axis=-1)

    # Indices sorted in decreasing order
    sorted_indices = mx.argsort(-logits).squeeze(0)
    sorted_probs = probs[..., sorted_indices]

    # Top probability
    top_probs = probs[..., sorted_indices[0]]

    # Calculate the min_p threshold
    scaled_min_p = min_p * top_probs

    # Mask tokens that have a probability less than the scaled min_p
    tokens_to_remove = sorted_probs < scaled_min_p
    tokens_to_remove[..., :min_tokens_to_keep] = False

    # Create pool of tokens with probability less than scaled min_p
    selected_probs = mx.where(tokens_to_remove, 0, sorted_probs)

    # Return sampled token
    sorted_token = mx.random.categorical(mx.log(selected_probs))
    return sorted_indices[sorted_token]


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def top_p_sampling(logits: mx.array, top_p: float, temperature: float) -> mx.array:
    """
    Apply top-p (nucleus) sampling to logits.

    Args:
        logits: The logits from the model's output.
        top_p: The cumulative probability threshold for top-p filtering.
        temperature: Temperature parameter for softmax distribution reshaping.
    Returns:
        token selected based on the top-p criterion.
    """
    # referenced implementation from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L449-L460
    probs = mx.softmax(logits * (1 / temperature), axis=-1)

    # sort probs in ascending order
    sorted_indices = mx.argsort(probs, axis=-1)
    sorted_probs = probs[..., sorted_indices.squeeze(0)]

    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    # select tokens with cumulative probs below threshold
    top_probs = mx.where(
        cumulative_probs > 1 - top_p,
        sorted_probs,
        0,
    )

    sorted_token = mx.random.categorical(mx.log(top_probs))
    token = sorted_indices.squeeze(0)[sorted_token]

    return token


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def categorical_sampling(logits, temp):
    return mx.random.categorical(logits * (1 / temp))
