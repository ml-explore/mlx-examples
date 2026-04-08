"""Training script for a masked diffusion language model (LLaDA-style).

Usage (pre-training from scratch on WikiText-103):
    python train.py

Usage (fine-tuning / SFT on instruction data):
    python train.py --sft --data path/to/sft_data.jsonl

The training data file should be a JSONL where each line is:
  {"text": "..."} for pre-training
  {"prompt": "...", "response": "..."} for SFT

For a quick smoke-test with a tiny model:
    python train.py --d_model 256 --n_layers 4 --n_heads 4 --mlp_hidden_size 512 \
        --batch_size 2 --iters 100
"""

import argparse
import json
import math
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map

from model import Model, ModelArgs


# ---------------------------------------------------------------------------
# Masking / forward process
# ---------------------------------------------------------------------------


def forward_process(
    input_ids: mx.array,
    mask_token_id: int,
    eps: float = 1e-3,
) -> tuple[mx.array, mx.array, mx.array]:
    """Apply random token masking for masked diffusion training.

    For each sequence in the batch, samples a masking probability t ~ U(eps, 1)
    and independently masks each token with probability t.

    Args:
        input_ids: Integer tensor of shape (B, L).
        mask_token_id: Token ID used as the [MASK] placeholder.
        eps: Minimum masking probability (avoids t=0).

    Returns:
        noisy_ids: Masked input ids of shape (B, L).
        masked: Boolean mask of shape (B, L); True where tokens were replaced.
        p_mask: Masking probability per token of shape (B, L).
    """
    B, L = input_ids.shape
    # Sample one masking probability per sequence
    t = mx.random.uniform(shape=(B,), low=eps, high=1.0)
    p_mask = mx.broadcast_to(t[:, None], (B, L))

    # Create binary mask by comparing uniform samples to p_mask
    u = mx.random.uniform(shape=(B, L))
    masked = u < p_mask

    noisy_ids = mx.where(masked, mask_token_id, input_ids)
    return noisy_ids, masked, p_mask


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def pretrain_loss(model: Model, input_ids: mx.array) -> mx.array:
    """Masked diffusion pre-training loss.

    Cross-entropy at masked positions, divided by the masking probability,
    then averaged over all token positions.  This is an upper bound on the
    negative log-likelihood of the model distribution.

    Args:
        model: The mask-predictor network.
        input_ids: Clean token ids of shape (B, L).

    Returns:
        Scalar loss.
    """
    mask_token_id = model.args.mask_token_id
    noisy_ids, masked, p_mask = forward_process(input_ids, mask_token_id)

    logits = model(noisy_ids)  # (B, L, V)

    # Cross-entropy loss for every token
    token_ce = nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        input_ids.reshape(-1),
        reduction="none",
    )  # (B*L,)
    token_ce = token_ce.reshape(input_ids.shape)  # (B, L)

    # Weight each token's loss by 1/p_mask (inverse masking probability)
    # Only sum at masked positions; un-masked positions have zero gradient
    weighted = token_ce * masked / p_mask

    # Normalise by total tokens (masked + unmasked), as in the paper
    B, L = input_ids.shape
    return weighted.sum() / (B * L)


def sft_loss(
    model: Model,
    input_ids: mx.array,
    prompt_lengths: mx.array,
) -> mx.array:
    """Supervised fine-tuning loss for conditional masked diffusion.

    Masking is applied only to the *response* portion of each sequence;
    the prompt tokens remain intact.  Loss is normalised by the length of
    the response (not the full sequence) as described in the LLaDA paper.

    Args:
        model: The mask-predictor network.
        input_ids: Token ids of shape (B, L) containing both prompt and response.
        prompt_lengths: Integer tensor of shape (B,) with each prompt's length.

    Returns:
        Scalar loss.
    """
    mask_token_id = model.args.mask_token_id
    B, L = input_ids.shape

    noisy_ids, masked, p_mask = forward_process(input_ids, mask_token_id)

    # Restore prompt tokens — do not mask the conditioning input
    positions = mx.broadcast_to(mx.arange(L)[None, :], (B, L))
    prompt_region = positions < prompt_lengths[:, None]
    noisy_ids = mx.where(prompt_region, input_ids, noisy_ids)

    logits = model(noisy_ids)  # (B, L, V)

    token_ce = nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        input_ids.reshape(-1),
        reduction="none",
    ).reshape(input_ids.shape)

    # Only apply loss where tokens were masked (response region)
    masked = (noisy_ids == mask_token_id)
    weighted = token_ce * masked / p_mask

    # Normalise by answer (response) length per sequence
    response_len = (L - prompt_lengths).astype(mx.float32)  # (B,)
    per_seq = weighted.sum(axis=1) / mx.maximum(response_len, 1.0)
    return per_seq.mean()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def tokenize_file(path: str, tokenizer, max_length: int) -> list[list[int]]:
    """Tokenise a JSONL file into fixed-length token id lists."""
    samples = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("text", obj.get("prompt", "") + obj.get("response", ""))
            ids = tokenizer.encode(text)
            # Chunk into max_length windows
            for start in range(0, len(ids), max_length):
                chunk = ids[start : start + max_length]
                if len(chunk) == max_length:
                    samples.append(chunk)
    return samples


def iterate_batches(
    data: list[list[int]],
    batch_size: int,
    shuffle: bool = True,
):
    """Yield batches of token id arrays."""
    indices = np.arange(len(data))
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, len(indices) - batch_size + 1, batch_size):
            batch_idx = indices[start : start + batch_size]
            batch = mx.array([data[i] for i in batch_idx])
            yield batch


def iterate_sft_batches(
    data: list[dict],
    tokenizer,
    batch_size: int,
    max_length: int,
    shuffle: bool = True,
):
    """Yield batches for SFT, returning (input_ids, prompt_lengths)."""
    records = []
    for obj in data:
        prompt_ids = tokenizer.encode(obj["prompt"])
        response_ids = tokenizer.encode(obj["response"])
        ids = prompt_ids + response_ids
        if len(ids) > max_length:
            ids = ids[:max_length]
        padded = ids + [tokenizer.eos_token_id] * (max_length - len(ids))
        records.append({"ids": padded, "prompt_len": len(prompt_ids)})

    indices = np.arange(len(records))
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, len(indices) - batch_size + 1, batch_size):
            batch_idx = indices[start : start + batch_size]
            batch = [records[i] for i in batch_idx]
            input_ids = mx.array([r["ids"] for r in batch])
            prompt_lengths = mx.array([r["prompt_len"] for r in batch])
            yield input_ids, prompt_lengths


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def cosine_schedule(
    step: int,
    total_steps: int,
    warmup_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser(description="Train a masked diffusion LM")

    # Model
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_kv_heads", type=int, default=8)
    parser.add_argument("--mlp_hidden_size", type=int, default=2048)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--mask_token_id", type=int, default=32000)

    # Training
    parser.add_argument("--sft", action="store_true", help="SFT mode")
    parser.add_argument("--data", type=str, default=None, help="Path to JSONL data")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Checkpointing
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)

    # Logging
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    mx.random.seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ model
    model_args = ModelArgs(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        mlp_hidden_size=args.mlp_hidden_size,
        vocab_size=args.vocab_size + 1,  # +1 for mask token
        mask_token_id=args.mask_token_id,
    )
    model = Model(model_args)
    mx.eval(model.parameters())

    n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Model parameters: {n_params / 1e6:.1f}M")

    # Save config alongside checkpoints
    config_path = save_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(model_args), f, indent=2)

    # --------------------------------------------------------------- resume
    start_step = 0
    if args.resume:
        weights = mx.load(args.resume)
        model.load_weights(list(weights.items()))
        mx.eval(model.parameters())
        # Infer step from filename if possible
        try:
            start_step = int(Path(args.resume).stem.split("_")[-1])
        except ValueError:
            pass
        print(f"Resumed from {args.resume} at step {start_step}")

    # --------------------------------------------------------------- data
    if args.data:
        # Load tokenizer if data path provided
        from transformers import AutoTokenizer

        tok_name = args.tokenizer or "meta-llama/Meta-Llama-3-8B"
        tokenizer = AutoTokenizer.from_pretrained(tok_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        raw = [json.loads(l) for l in open(args.data)]
        if args.sft:
            data_iter = iterate_sft_batches(
                raw, tokenizer, args.batch_size, args.max_length
            )
        else:
            samples = tokenize_file(args.data, tokenizer, args.max_length)
            data_iter = iterate_batches(samples, args.batch_size)
    else:
        # Synthetic random data for quick testing
        print("No data file specified — using random synthetic token ids.")
        rng = np.random.default_rng(args.seed)

        def _random_iter():
            while True:
                ids = rng.integers(
                    0,
                    args.vocab_size,
                    size=(args.batch_size, args.max_length),
                )
                yield mx.array(ids)

        data_iter = _random_iter()
        args.sft = False  # Can't do SFT without real data

    # ------------------------------------------------------------ optimizer
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.1)

    # --------------------------------------------------------- training step
    def loss_and_grad(model, batch, sft=False):
        if sft:
            input_ids, prompt_lengths = batch
            return sft_loss(model, input_ids, prompt_lengths)
        else:
            return pretrain_loss(model, batch)

    loss_and_grad_fn = nn.value_and_grad(model, loss_and_grad)

    max_norm = args.grad_clip

    def clip_gradients(grads):
        leaves = [g for _, g in tree_flatten(grads) if isinstance(g, mx.array)]
        total_norm = mx.sqrt(sum(mx.sum(g**2) for g in leaves))
        scale = mx.minimum(max_norm / (total_norm + 1e-6), 1.0)
        return tree_map(lambda g: g * scale if isinstance(g, mx.array) else g, grads)

    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(batch, sft=False):
        loss, grads = loss_and_grad_fn(model, batch, sft)
        grads = clip_gradients(grads)
        optimizer.update(model, grads)
        return loss

    # ----------------------------------------------------------------- loop
    losses = []
    t_start = time.perf_counter()

    for step in range(start_step, args.iters):
        # Update learning rate
        lr = cosine_schedule(step, args.iters, args.warmup, args.lr, args.min_lr)
        optimizer.learning_rate = lr

        batch = next(data_iter)
        loss = train_step(batch, args.sft)
        mx.eval(state)
        losses.append(loss.item())

        if (step + 1) % args.log_every == 0:
            elapsed = time.perf_counter() - t_start
            avg_loss = sum(losses[-args.log_every :]) / args.log_every
            tps = args.log_every / elapsed
            print(
                f"step {step + 1:6d} | loss {avg_loss:.4f} | lr {lr:.2e} | {tps:.1f} it/s"
            )
            t_start = time.perf_counter()

        if (step + 1) % args.save_every == 0:
            ckpt = save_dir / f"weights_{step + 1:07d}.safetensors"
            flat = dict(tree_flatten(model.parameters()))
            mx.save_safetensors(str(ckpt), flat)
            print(f"Saved checkpoint → {ckpt}")

    # Final checkpoint
    ckpt = save_dir / f"weights_{args.iters:07d}.safetensors"
    flat = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(ckpt), flat)
    print(f"Training complete. Final checkpoint → {ckpt}")


if __name__ == "__main__":
    main()
