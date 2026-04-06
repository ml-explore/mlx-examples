import argparse
import math
import os
import time
from functools import partial
from pathlib import Path

import datasets
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# import evaluate
import utils
import wandb
from mlx.nn.utils import average_gradients
from mlx.utils import tree_map_with_path, tree_reduce


def load_data(tokenizer, data_path="allenai/dolma3_mix-150B-1025", valid_size=1000):
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

    def tokenize(d):
        tokens = tokenizer.encode(d["text"], add_special_tokens=False)
        tokens.append(tokenizer.eos_token_id)
        return {"data": tokens}

    ds = ds.map(tokenize)
    local_valid_size = valid_size // size
    valid_ds = ds.take(local_valid_size)
    train_ds = ds.skip(local_valid_size)
    return train_ds, valid_ds


def iterate_batches(dataset, context_size, batch_size):
    """
    Simply concatenate documents until the batch is full.
    """
    dataset = iter(dataset)
    seq_len = context_size + 1
    batch = np.empty((batch_size * seq_len), np.int32)
    d_next = []
    while True:
        i = 0
        while i < len(batch):
            if len(d_next) > 0:
                d = d_next
                d_next = []
            else:
                d = next(dataset, None)
                if d is None:
                    break
                d = d["data"]
            e = i + len(d)
            if e > len(batch):
                trim = e - len(batch)
                d_next = d[-trim:]
                d = d[:-trim]
                e = len(batch)
            batch[i:e] = d
            i += len(d)
        # Iterator ended
        if i < len(batch):
            break
        yield batch.reshape(batch_size, seq_len)


def main(config, save_dir):

    np.random.seed(config.seed)
    mx.random.seed(config.seed)

    rank, world_size = utils.init_distributed()
    batch_size = config.batch_size
    context_size = config.context_size

    # data.download_eval_bundle()

    optimizer = utils.load_optimizer(config)
    tokenizer = utils.load_tokenizer()
    train_set, valid_set = load_data(tokenizer)

    model = utils.load_model(config.model)
    dtype = getattr(mx, config.data_type)

    # Quantize the model if specified in the config
    quant_params = set()
    if quant := config.model.get("quantization", False):

        def class_predicate(p, m):
            if isinstance(m, nn.Linear):
                quant_params.add(p + ".weight")
                quant_params.add(p + ".scales")
                return True
            return False

        nn.quantize(
            model,
            mode=quant["mode"],
            quantize_input=True,
            class_predicate=class_predicate,
        )

    @mx.compile
    def loss_fn(params, sample):
        model.update(
            tree_map_with_path(
                lambda p, x: x.astype(dtype) if p not in quant_params else x, params
            )
        )
        inputs = sample[:, :-1]
        targets = sample[:, 1:]

        logits = model(inputs).astype(mx.float32)
        losses = nn.losses.cross_entropy(logits, targets, reduction="none")
        return losses.sum() / targets.size

    state = [optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(sample, params):
        loss, grads = mx.value_and_grad(loss_fn)(params, sample)
        grads = average_gradients(grads, all_reduce_size=4e9)
        grads, grad_norm = optim.clip_grad_norm(grads, max_norm=config.max_grad_norm)
        params = optimizer.apply_gradients(grads, params)
        return loss, grad_norm, params

    def eval_fn(params, dataset):
        data_it = iterate_batches(
            dataset,
            context_size=context_size,
            batch_size=batch_size,
        )
        losses = 0
        ntoks = 0
        toks_per_batch = context_size * batch_size
        for sample in data_it:
            loss = loss_fn(params, mx.array(sample))
            losses += loss * toks_per_batch
            mx.eval(losses)
            ntoks += toks_per_batch
        return losses / ntoks

    params = model.trainable_parameters()
    nparams = tree_reduce(lambda acc, p: acc + p.size, params, 0)
    if rank == 0:
        print(f"Model has {nparams} parameters.")
    mx.eval(params)

    train_iterator = iterate_batches(
        train_set,
        context_size=config.context_size,
        batch_size=config.batch_size,
    )

    metrics = utils.Metrics()
    tokens = 0
    tic = time.perf_counter()
    for it, sample in zip(range(0, config.num_steps), train_iterator):
        loss, grad_norm, params = step(mx.array(sample), params)
        loss = mx.distributed.all_sum(loss) / world_size
        grad_norm = mx.distributed.all_sum(grad_norm) / world_size
        mx.eval(loss, grad_norm, params, state)
        metrics.train_loss.append(loss.item())
        metrics.grad_norm.append(grad_norm.item())
        tokens += config.context_size * config.batch_size * world_size

        if (it + 1) % config.steps_per_report == 0:
            toc = time.perf_counter()
            metrics.step = it + 1
            metrics.tokens += tokens
            metrics.its_per_sec = config.steps_per_report * world_size / (toc - tic)
            metrics.toks_per_sec = tokens / (toc - tic)
            tokens = 0

            if (it + 1) % config.steps_per_eval == 0:
                # Do the evaluation in the final precision, but only cast the model once
                model.update(params)
                model.eval()
                model.set_dtype(dtype)
                eval_params = model.parameters()
                mx.eval(eval_params)
                loss = eval_fn(eval_params, valid_set)
                loss = mx.distributed.all_sum(loss) / world_size
                metrics.valid_loss = loss.item()
                metrics.valid_ppl = math.exp(metrics.valid_loss)
                model.train()
                model.update(params)

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
    parser = argparse.ArgumentParser(description="Train an LM.")
    parser.add_argument(
        "--config",
        default="configs/base_600m.yaml",
        type=str,
        help="Experiment config",
    )
    parser.add_argument(
        "--save-dir",
        default="checkpoints",
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
    config = utils.load_config(args.config)
    if mx.distributed.init().rank() == 0:
        wandb_kwargs = dict(
            project="flash_lm",
            name=args.wandb_name,
            tags=["pretrain", config.model["model_type"]],
        )
        if args.wandb_name is None:
            wandb_kwargs["mode"] = "disabled"
        run = wandb.init(**wandb_kwargs)
    main(config, args.save_dir)
