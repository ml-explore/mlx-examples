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
        metadata={"help": "Temperature parameter for ORPO training."}
    )
    reward_scaling: float = field(
        default=1.0,
        metadata={"help": "Reward scaling factor for ORPO training, not implemented."}
    )


def orpo_loss(model, chosen, rejected, chosen_masks, rejected_masks, preference_scores, beta=0.1):
    def get_logps(model, x, mask):
        inputs = x[:, :-1]
        targets = x[:, 1:]
        logits = model(inputs)
        logp = -nn.losses.cross_entropy(logits, targets, reduction='none')
        seq_lengths = mask[:, :-1].sum(-1)
        logp_sum = (logp * mask[:, :-1]).sum(-1) / seq_lengths
        logits_mean = (logits * mask[:, :-1, None]).sum() / mask[:, :-1].sum()
        return logp_sum, logits_mean

    policy_chosen_logps, chosen_logits_mean = get_logps(model, chosen, chosen_masks)
    policy_rejected_logps, rejected_logits_mean = get_logps(model, rejected, rejected_masks)

    policy_chosen_logps = policy_chosen_logps * preference_scores

    log_odds = (policy_chosen_logps - policy_rejected_logps) - (
        mx.log1p(-mx.exp(policy_chosen_logps)) - mx.log1p(-mx.exp(policy_rejected_logps))
    )

    ratio = nn.log_sigmoid(log_odds)
    loss = -beta * ratio

    chosen_reward = beta * policy_chosen_logps
    rejected_reward = beta * policy_rejected_logps
    reward = mx.stack([mx.mean(chosen_reward), mx.mean(rejected_reward)])

    num_tokens = chosen_masks.sum() + rejected_masks.sum()

    metrics = {
        'accuracies': mx.mean((chosen_reward > rejected_reward).astype(mx.float32)),
        'margins': mx.mean(chosen_reward - rejected_reward),
        'policy_rejected_logps': mx.mean(policy_rejected_logps),
        'policy_chosen_logps': mx.mean(policy_chosen_logps),
        'rejected_logits_mean': mx.mean(rejected_logits_mean),
        'chosen_logits_mean': mx.mean(chosen_logits_mean)
    }

    return mx.mean(loss), reward, num_tokens, metrics


def iterate_orpo_batches(dataset, batch_size, max_seq_length, train=False):
    """Batch iterator for ORPO with preference scores"""
    idx = sorted(range(len(dataset)), key=lambda idx: len(dataset[idx]['chosen']))
    
    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size}"
            f" examples but only has {len(dataset)}."
        )
    
    step = mx.distributed.init().size()
    if batch_size % step != 0:
        raise ValueError("Batch size must be divisible by number of workers")
    
    batch_idx = [idx[i:i + batch_size:step] for i in range(0, len(idx) - batch_size + 1, batch_size)]
    
    while True:
        indices = np.random.permutation(len(batch_idx)) if train else range(len(batch_idx))
        for i in indices:
            batch = [dataset[j] for j in batch_idx[i]]
            
            chosen_lengths = [len(x['chosen']) for x in batch]
            rejected_lengths = [len(x['rejected']) for x in batch]
            max_length = min(max(max(chosen_lengths), max(rejected_lengths)), max_seq_length)
            pad_to = 8
            max_length_in_batch = pad_to * ((max_length + pad_to - 1) // pad_to)
            
            batch_size_per_device = batch_size // step
            chosen_arr = np.zeros((batch_size_per_device, max_length_in_batch), np.int32)
            rejected_arr = np.zeros((batch_size_per_device, max_length_in_batch), np.int32)
            chosen_masks = np.zeros((batch_size_per_device, max_length_in_batch), np.float32)
            rejected_masks = np.zeros((batch_size_per_device, max_length_in_batch), np.float32)
            
            preference_scores = np.array([x.get('preference_score', 1.0) for x in batch], np.float32)
            
            for j in range(batch_size_per_device):
                chosen_length = min(chosen_lengths[j], max_length_in_batch)
                rejected_length = min(rejected_lengths[j], max_length_in_batch)
                
                chosen_arr[j, :chosen_length] = batch[j]['chosen'][:chosen_length]
                chosen_masks[j, :chosen_length] = 1.0
                rejected_arr[j, :rejected_length] = batch[j]['rejected'][:rejected_length]
                rejected_masks[j, :rejected_length] = 1.0
            
            yield (
                mx.array(chosen_arr),
                mx.array(rejected_arr),
                mx.array(chosen_masks),
                mx.array(rejected_masks),
                mx.array(preference_scores)
            )
            
        if not train:
            break


def evaluate_orpo(model, dataset, batch_size, num_batches, beta: float, max_seq_length=2048):
    all_losses = 0
    all_rewards = mx.zeros((2,))
    all_metrics = None
    ntokens = 0
    
    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)
    for _, batch in zip(
        index_iterator,
        iterate_orpo_batches(
            dataset=dataset,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        ),
    ):
        chosen, rejected, chosen_masks, rejected_masks, preference_scores = batch
        lvalue, reward, toks, metrics = orpo_loss(
            model=model,
            chosen=chosen,
            rejected=rejected,
            chosen_masks=chosen_masks,
            rejected_masks=rejected_masks,
            preference_scores=preference_scores,
            beta=beta
        )
        all_losses += lvalue * toks
        all_rewards += reward * toks
        ntokens += toks
        
        if all_metrics is None:
            all_metrics = {k: v * toks for k, v in metrics.items()}
        else:
            for k, v in metrics.items():
                all_metrics[k] += v * toks

    mx.eval(all_losses, all_rewards, ntokens)
    all_losses = mx.distributed.all_sum(all_losses)
    all_rewards = mx.distributed.all_sum(all_rewards)
    ntokens = mx.distributed.all_sum(ntokens)
    all_metrics = {k: mx.distributed.all_sum(v) for k, v in all_metrics.items()}
    
    avg_metrics = {k: (v / ntokens).item() for k, v in all_metrics.items()}
    avg_rewards = (all_rewards / ntokens).tolist()
    avg_loss = (all_losses / ntokens).item()
    
    return avg_loss, avg_rewards, ntokens, avg_metrics


def train_orpo(
    model,
    tokenizer,
    optimizer,
    train_dataset,
    val_dataset,
    loss: callable = orpo_loss,
    args: ORPOTrainingArgs = ORPOTrainingArgs(),
    training_callback: TrainingCallback = None,
):
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
        chosen, rejected, chosen_masks, rejected_masks, preference_scores = batch
        
        (lvalue, reward, toks, metrics), grad = loss_value_and_grad(
            model, 
            chosen, 
            rejected, 
            chosen_masks, 
            rejected_masks,
            preference_scores=preference_scores,
        )

        grad = average_gradients(grad)
        optimizer.update(model, grad)

        return lvalue, reward, toks, metrics

    def loss_wrapper(model, chosen, rejected, chosen_masks, rejected_masks, preference_scores):
        return loss(
            model=model,
            chosen=chosen,
            rejected=rejected,
            chosen_masks=chosen_masks,
            rejected_masks=rejected_masks,
            preference_scores=preference_scores,
            beta=args.beta
        )
    
    loss_value_and_grad = nn.value_and_grad(model, loss_wrapper)

    # Training loop with progress tracking
    losses = 0
    rewards = mx.zeros((2,))
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    accumulated_metrics = {
        'accuracies': 0,
        'margins': 0,
        'policy_rejected_logps': 0,
        'policy_chosen_logps': 0,
        'rejected_logits_mean': 0,
        'chosen_logits_mean': 0
    }
    
    start = time.perf_counter()
    for it, batch in zip(
        range(1, args.iters + 1),
        iterate_orpo_batches(
            dataset=train_dataset,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            train=True,
        ),
    ):
        if it == 1 or it % args.steps_per_eval == 0 or it == args.iters:
            stop = time.perf_counter()
            val_loss, val_rewards, val_ntokens, val_metrics = evaluate_orpo(
                model=model,
                dataset=val_dataset,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                beta=args.beta
            )
            val_time = time.perf_counter() - stop
            if rank == 0:
                print(
                    f"Iter {it}: "
                    f"Val loss {val_loss:.3f}, "
                    f"Val chosen reward {val_rewards[0]:.3f}, "
                    f"Val rejected reward {val_rewards[1]:.3f}, "
                    f"Val accuracy {val_metrics['accuracies']:.3f}, "
                    f"Val margin {val_metrics['margins']:.3f}, "
                    f"Val took {val_time:.3f}s",
                    flush=True,
                )

            if training_callback is not None:
                training_callback.on_val_loss_report({
                    "iteration": it,
                    "val_loss": val_loss,
                    "val_chosen_reward": val_rewards[0],
                    "val_rejected_reward": val_rewards[1],
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    "val_time": val_time,
                })

            start = time.perf_counter()

        # Training step
        lvalue, reward, toks, metrics = step(batch)
        losses += lvalue
        rewards += reward
        n_tokens += toks
        steps += 1

        for k, v in metrics.items():
            accumulated_metrics[k] += v

        mx.eval(state, losses, rewards, n_tokens)

        if it % args.steps_per_report == 0 or it == args.iters:
            stop = time.perf_counter()

            train_loss = mx.distributed.all_sum(losses).item() / (steps * world_size)
            train_rewards = [r / (steps * world_size) for r in mx.distributed.all_sum(rewards).tolist()]
            avg_metrics = {k: v / (steps * world_size) for k, v in accumulated_metrics.items()}
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
                    f"Accuracy {avg_metrics['accuracies']:.3f}, "
                    f"Margin {avg_metrics['margins']:.3f}, "
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
                    "train_chosen_reward": train_rewards[0],
                    "train_rejected_reward": train_rewards[1],
                    **{f"train_{k}": v for k, v in avg_metrics.items()},
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
            accumulated_metrics = {k: 0 for k in accumulated_metrics}
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