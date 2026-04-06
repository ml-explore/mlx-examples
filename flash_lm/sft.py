import argparse
import math
import os
import random
import time
from functools import partial
from pathlib import Path

import datasets
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import utils
import wandb
from mlx.nn.utils import average_gradients
from mlx.utils import tree_map, tree_reduce


def buffer_batches(dataset, batch_size):

    def _sort_batch_shuffle(buffer):
        buffer.sort(key=lambda x: len(x[0]))
        batches = [
            buffer[s : s + batch_size] for s in range(0, len(buffer), batch_size)
        ]
        random.shuffle(batches)
        return batches

    buffer_size = batch_size * 1000
    buffer = []
    for d in dataset:
        buffer.append(d)
        if len(buffer) >= buffer_size:
            for b in _sort_batch_shuffle(buffer):
                yield b
            buffer = []

    for b in _sort_batch_shuffle(buffer):
        yield b


def iterate_batches(dataset, batch_size, max_length=None):
    """
    Add full documents into the batch with padding based on the maximum
    document length.
    """

    def _collate(seqs, length, dtype):
        return np.array([s + [0] * (length - len(s)) for s in seqs], dtype)

    def _round_up(n):
        m = 512
        n = m * ((n + m - 1) // m)
        if max_length is not None:
            n = min(n, max_length)
        return n + 1

    dataset = ((d["tokens"], d["mask"]) for d in dataset)
    for batch in buffer_batches(dataset, batch_size):
        tokens, masks = zip(*batch)
        if max_length is not None:
            tokens = [t[: max_length + 1] for t in tokens]
            masks = [m[: max_length + 1] for m in masks]
        lengths = [len(t) for t in tokens]
        length = _round_up(max(lengths))
        yield {
            "data": _collate(tokens, length=length, dtype=np.int32),
            "mask": _collate(masks, length=length, dtype=bool),
            "lengths": lengths,
        }


def load_data(tokenizer, data_path="allenai/Dolci-Instruct-SFT", valid_size=1000):
    group = mx.distributed.init()
    size = group.size()
    rank = group.rank()
    ds = datasets.load_dataset(
        data_path,
        split="train",
        streaming=True,
    )
    ds = ds.shard(num_shards=size, index=rank)
    ds = ds.shuffle(buffer_size=10000)

    # Tokenize data so that only the assistant generations are not masked.
    n_mask = len(tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False))

    def tokenize(d):
        tokens = []
        mask = []
        for m in d["messages"]:
            role = m.get("role")
            if role == "user":
                local_tokens = tokenizer.apply_chat_template([m], return_dict=False)
                mask.extend([False] * len(local_tokens))
            elif role == "assistant":
                local_tokens = tokenizer.apply_chat_template([m], return_dict=False)
                mask.extend([False] * n_mask)
                mask.extend([True] * (len(local_tokens) - n_mask))
            else:
                raise ValueError(f"Unknown role {role}")
            tokens.extend(local_tokens)
        return {"tokens": tokens, "mask": mask}

    ds = ds.map(tokenize)
    local_valid_size = valid_size // size
    valid_ds = ds.take(local_valid_size)
    train_ds = ds.skip(local_valid_size)
    return train_ds, valid_ds


def main(config, checkpoint_dir, save_dir):
    random.seed(config.seed)
    np.random.seed(config.seed)
    mx.random.seed(config.seed)

    rank, world_size = utils.init_distributed()
    batch_size = config.batch_size
    max_length = config.context_size

    optimizer = utils.load_optimizer(config)
    tokenizer = utils.load_tokenizer()
    train_set, valid_set = load_data(tokenizer)

    model = utils.load_model(config.model)
    model.load_weights(str(Path(checkpoint_dir) / "model.safetensors"))

    dtype = getattr(mx, config.data_type)

    def to_mlx(sample):
        return {k: mx.array(v) for k, v in sample.items()}

    def loss_fn(params, sample):
        model.update(tree_map(lambda x: x.astype(dtype), params))
        inputs = sample["data"][:, :-1]
        targets = sample["data"][:, 1:]

        logits = model(inputs).astype(mx.float32)
        losses = nn.losses.cross_entropy(logits, targets, reduction="none")
        mask = sample["mask"][:, 1:]
        # Avoid divide by 0 (loss should be 0)
        ntoks = mx.maximum(mask.sum(), 1)
        loss = (losses * mask).sum() / ntoks
        return loss, ntoks

    state = [optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(sample, params):
        (loss, ntoks), grads = mx.value_and_grad(loss_fn)(params, sample)
        grads = average_gradients(grads, all_reduce_size=4e9)
        grads, grad_norm = optim.clip_grad_norm(grads, max_norm=config.max_grad_norm)
        params = optimizer.apply_gradients(grads, params)
        return loss, ntoks, grad_norm, params

    def eval_fn(params):
        data_it = data.iterate_batches(
            valid_set,
            max_length=max_length,
            batch_size=batch_size,
        )
        losses = 0
        num_toks = 0
        toks_per_batch = max_length * batch_size
        for sample in data.prefetch(data_it):
            loss, ntoks = loss_fn(params, to_mlx(sample))
            losses += loss * ntoks
            num_toks += ntoks
            mx.eval(losses, num_toks)
        return losses / num_toks

    params = model.trainable_parameters()
    nparams = tree_reduce(lambda acc, p: acc + p.size, params, 0)
    if rank == 0:
        print(f"Model has {nparams} parameters.")

    mx.eval(params)

    train_iterator = iterate_batches(
        train_set,
        max_length=max_length,
        batch_size=config.batch_size,
    )

    metrics = utils.Metrics()
    tokens = 0
    tic = time.perf_counter()
    for it, sample in zip(range(0, config.num_steps), train_iterator):
        sample = to_mlx(sample)
        loss, _, grad_norm, params = step(sample, params)
        loss = mx.distributed.all_sum(loss) / world_size
        grad_norm = mx.distributed.all_sum(grad_norm) / world_size
        # Count all tokens processed without padding (not just loss tokens)
        num_tokens = mx.distributed.all_sum((sample["lengths"] - 1).sum())
        mx.eval(loss, num_tokens, grad_norm, params, state)
        metrics.train_loss.append(loss.item())
        metrics.grad_norm.append(grad_norm.item())
        tokens += num_tokens.item()

        if (it + 1) % config.steps_per_report == 0:
            toc = time.perf_counter()
            metrics.step = it + 1
            metrics.tokens += tokens
            metrics.its_per_sec = config.steps_per_report * world_size / (toc - tic)
            metrics.toks_per_sec = tokens / (toc - tic)
            tokens = 0

            if (it + 1) % config.steps_per_eval == 0:
                loss = eval_fn(params)
                loss = mx.distributed.all_sum(loss) / world_size
                metrics.valid_loss = loss.item()
                metrics.valid_ppl = math.exp(metrics.valid_loss)

            if rank == 0:
                if (it + 1) % config.steps_per_checkpoint == 0:
                    utils.save_checkpoint(save_dir, it, params, optimizer, config)
                    utils.save_checkpoint(save_dir, None, params, optimizer, config)
                utils.log_metrics(metrics)
            metrics = utils.Metrics(tokens=metrics.tokens)
            tic = time.perf_counter()

    if rank == 0:
        utils.save_checkpoint(save_dir, None, params, optimizer, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervise fine-tune an LM.")
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        type=str,
        help="Location to load the pretrained model checkpoint",
    )
    parser.add_argument(
        "--save-dir",
        default="sft_checkpoints",
        type=str,
        help="Location to save the model and checkpoints",
    )
    parser.add_argument(
        "--wandb-name",
        default=None,
        type=str,
        help="Name of experiment for wandb",
    )
    args = parser.parse_args()
    config = utils.load_config(args.checkpoint_dir)
    if mx.distributed.init().rank() == 0:
        wandb_kwargs = dict(
            project="flash_lm",
            name=args.wandb_name,
            tags=["pretrain", config.model["model_type"]],
        )
        if args.wandb_name is None:
            wandb_kwargs["mode"] = "disabled"
        run = wandb.init(**wandb_kwargs)
    main(config, args.checkpoint_dir, args.save_dir)
