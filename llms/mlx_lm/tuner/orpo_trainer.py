# Copyright © 2024 Apple Inc.

from dataclasses import dataclass, field
import mlx.core as mx
import mlx.nn as nn

from dpo_trainer import DPOTrainingArgs, iterate_dpo_batches, train_dpo, TrainingCallback


@dataclass
class ORPOTrainingArgs(DPOTrainingArgs):
    """
    Training arguments specific to ORPO, extending DPO arguments.
    """
    mu: float = field(
        default=0.5,
        metadata={"help": "ORPO KL divergence weight parameter"}
    )


def orpo_loss(
    model,
    reference_teacher_model,
    chosen: mx.array,
    rejected: mx.array,
    chosen_masks: mx.array,
    rejected_masks: mx.array,
    beta: float,
    delta: float,
    mu: float = 0.5,  # ORPO hyperparameter for balancing KL divergence
    loss_type: str = "sigmoid",
    is_reference_free: bool = False
):
    """
    Calculate ORPO loss for inputs.
    ORPO extends DPO by adding a KL regularization term to prevent overfitting to preferences.
    
    Args:
        model: Policy model
        reference_teacher_model: Reference model
        chosen: Chosen sequence tokens
        rejected: Rejected sequence tokens
        chosen_masks: Masks for chosen sequences
        rejected_masks: Masks for rejected sequences
        beta: Temperature parameter
        delta: Margin for DPOP loss type
        mu: ORPO hyperparameter for balancing KL divergence (default: 0.5)
        loss_type: Loss type ('sigmoid', 'hinge', 'ipo', or 'dpop')
        is_reference_free: Whether to use reference-free training
    Returns:
        Tuple of (loss, reward, num_tokens)
    """
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
    
    # Calculate reference model scores
    if not is_reference_free:
        reference_chosen_scores = mx.stop_gradient(make_predictions(reference_teacher_model, chosen, chosen_masks))
        reference_rejected_scores = mx.stop_gradient(make_predictions(reference_teacher_model, rejected, rejected_masks))
    else:
        reference_chosen_scores = mx.zeros_like(policy_chosen_scores)
        reference_rejected_scores = mx.zeros_like(policy_rejected_scores)

    # Compute average log probabilities if using IPO loss
    if loss_type == "ipo":
        policy_chosen_score = policy_chosen_scores.sum(-1) / num_chosen_tokens
        policy_rejected_score = policy_rejected_scores.sum(-1) / num_rejected_tokens
        reference_chosen_score = reference_chosen_scores.sum(-1) / num_chosen_tokens
        reference_rejected_score = reference_rejected_scores.sum(-1) / num_rejected_tokens
    else:
        policy_chosen_score = policy_chosen_scores.sum(-1)
        policy_rejected_score = policy_rejected_scores.sum(-1)
        reference_chosen_score = reference_chosen_scores.sum(-1)
        reference_rejected_score = reference_rejected_scores.sum(-1)

    # Calculate preference logits
    logits = (policy_chosen_score - policy_rejected_score) - (reference_chosen_score - reference_rejected_score)

    # Calculate preference loss based on loss type
    if loss_type == "sigmoid":
        preference_loss = -nn.log_sigmoid(beta * logits)
    elif loss_type == "hinge":
        preference_loss = nn.relu(1 - beta * logits)
    elif loss_type == "ipo":
        preference_loss = (logits - 1 / (2 * beta)) ** 2
    elif loss_type == "dpop":
        penalty = mx.maximum(mx.zeros_like(policy_chosen_score), reference_chosen_score - policy_chosen_score)
        preference_loss = -(nn.log_sigmoid(beta * logits) - delta * penalty)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Calculate KL divergence term for ORPO
    kl_div_chosen = mx.mean((policy_chosen_scores - reference_chosen_scores) ** 2)
    kl_div_rejected = mx.mean((policy_rejected_scores - reference_rejected_scores) ** 2)
    kl_regularization = mu * (kl_div_chosen + kl_div_rejected)

    # Combine preference loss and KL regularization
    loss = mx.mean(preference_loss) + kl_regularization
    
    num_tokens = (num_chosen_tokens + num_rejected_tokens).sum()

    # Calculate rewards for monitoring
    chosen_reward = beta * mx.mean(policy_chosen_score - reference_chosen_score)
    rejected_reward = beta * mx.mean(policy_rejected_score - reference_rejected_score)
    reward = mx.stack([chosen_reward, rejected_reward])

    return loss, reward, num_tokens


def evaluate_orpo(
    model,
    reference_model,
    dataset,
    tokenizer,
    batch_size,
    num_batches,
    beta: float,
    delta: float,
    mu: float = 0.5,
    max_seq_length=2048,
    loss_type="sigmoid",
    is_reference_free=False,
):
    """
    Evaluate model using ORPO metrics.
    
    Args:
        model: Policy model to evaluate
        reference_model: Reference model for comparison
        dataset: Evaluation dataset
        tokenizer: Tokenizer for processing text
        batch_size: Batch size for evaluation
        num_batches: Number of batches to evaluate (-1 for full dataset)
        beta: Temperature parameter
        delta: Margin for DPOP loss
        mu: ORPO KL divergence weight
        max_seq_length: Maximum sequence length
        loss_type: Type of loss function
        is_reference_free: Whether to use reference-free evaluation
    
    Returns:
        Tuple of (loss, rewards, kl_metrics), where:
        - loss is the total ORPO loss
        - rewards is [chosen_reward, rejected_reward]
        - kl_metrics is [chosen_kl, rejected_kl]
    """
    all_losses = 0
    all_rewards = mx.zeros((2,))  # [chosen_reward, rejected_reward]
    all_kl_divs = mx.zeros((2,))  # [chosen_kl, rejected_kl]
    ntokens = 0

    def compute_kl_divergence(policy_scores, reference_scores, masks):
        """Helper function to compute KL divergence metrics."""
        # Using MSE as a proxy for KL divergence as in the loss function
        valid_tokens = masks.sum()
        kl_div = ((policy_scores - reference_scores) ** 2 * masks).sum() / valid_tokens
        return kl_div

    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)

    for _, batch in zip(
        index_iterator,
        iterate_dpo_batches(  # Reusing DPO batch iterator
            dataset=dataset,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        ),
    ):
        chosen, rejected, chosen_masks, rejected_masks = batch
        
        # Get model predictions
        def make_predictions(model, x, mask):
            inputs = x[:, :-1]
            targets = x[:, 1:]
            logits = model(inputs)
            logits = logits.astype(mx.float32)
            return -nn.losses.cross_entropy(logits, targets) * mask[:, :-1]
        
        # Get scores for both models
        policy_chosen_scores = make_predictions(model, chosen, chosen_masks)
        policy_rejected_scores = make_predictions(model, rejected, rejected_masks)
        
        if not is_reference_free:
            reference_chosen_scores = mx.stop_gradient(
                make_predictions(reference_model, chosen, chosen_masks)
            )
            reference_rejected_scores = mx.stop_gradient(
                make_predictions(reference_model, rejected, rejected_masks)
            )
        else:
            reference_chosen_scores = mx.zeros_like(policy_chosen_scores)
            reference_rejected_scores = mx.zeros_like(policy_rejected_scores)

        # Compute KL divergences
        chosen_kl = compute_kl_divergence(
            policy_chosen_scores, reference_chosen_scores, chosen_masks[:, :-1]
        )
        rejected_kl = compute_kl_divergence(
            policy_rejected_scores, reference_rejected_scores, rejected_masks[:, :-1]
        )
        all_kl_divs += mx.stack([chosen_kl, rejected_kl])

        # Compute ORPO loss and rewards
        loss, reward, toks = orpo_loss(
            model=model,
            reference_teacher_model=reference_model,
            chosen=chosen,
            rejected=rejected,
            chosen_masks=chosen_masks,
            rejected_masks=rejected_masks,
            beta=beta,
            delta=delta,
            mu=mu,
            loss_type=loss_type,
            is_reference_free=is_reference_free,
        )
        
        all_losses += loss * toks
        all_rewards += reward
        ntokens += toks
        mx.eval(all_losses, all_rewards, all_kl_divs, ntokens)

    # Aggregate metrics across distributed workers if necessary
    all_losses = mx.distributed.all_sum(all_losses)
    all_rewards = mx.distributed.all_sum(all_rewards)
    all_kl_divs = mx.distributed.all_sum(all_kl_divs)
    ntokens = mx.distributed.all_sum(ntokens)

    # Normalize metrics
    avg_loss = (all_losses / ntokens).item()
    avg_rewards = [r / mx.distributed.init().size() for r in all_rewards.tolist()]
    avg_kl_divs = [kl / mx.distributed.init().size() for kl in all_kl_divs.tolist()]

    return avg_loss, avg_rewards, avg_kl_divs


def train_orpo(
    model,
    reference_model,
    tokenizer,
    optimizer,
    train_dataset,
    val_dataset,
    args: ORPOTrainingArgs = ORPOTrainingArgs(),
    training_callback: TrainingCallback = None,
):
    """
    Train a model using ORPO (Offline Rejection Preference Optimization).
    This function adapts the DPO training loop to use ORPO loss.
    """
    return train_dpo(
        model=model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        args=args,
        loss_fn=orpo_loss,
        training_callback=training_callback,
        loss_type=args.loss_type,
    )