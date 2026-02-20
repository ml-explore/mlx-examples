"""Masked-diffusion language model generation (inference).

Implements the iterative unmasking algorithm from LLaDA:
  "LLaDA: Large Language Diffusion with mAsking"
  https://arxiv.org/abs/2502.09992

Starting from a fully masked response region, the model predicts token
probabilities at every masked position simultaneously.  At each step, the most
confident predictions are "unmasked" (committed), while the rest are either kept
masked or re-masked (depending on the remasking strategy).  This is repeated
until all positions are unmasked.

Usage:
    # Load a pre-trained LLaDA-8B-Instruct model converted to MLX format:
    python generate.py \\
        --model-path path/to/mlx_llada \\
        --prompt "What is the capital of France?" \\
        --gen-length 128 \\
        --steps 128

    # Quick test with a tiny random model (no real weights):
    python generate.py --demo
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from model import Model, ModelArgs


# ---------------------------------------------------------------------------
# Gumbel-noise categorical sampling
# ---------------------------------------------------------------------------


def add_gumbel_noise(logits: mx.array, temperature: float) -> mx.array:
    """Apply Gumbel noise for stochastic categorical sampling.

    At temperature=0 this reduces to argmax (greedy decoding).
    Using higher temperature introduces diversity at the cost of coherence.

    Note: The LLaDA paper recommends float32 precision here; low-precision
    Gumbel noise slightly improves perplexity but hurts generation quality.

    Args:
        logits: Unnormalised log-probabilities of shape (..., vocab_size).
        temperature: Gumbel noise scale; 0 = greedy.

    Returns:
        Noisy logits in the same shape (for use with argmax).
    """
    if temperature == 0.0:
        return logits
    # Sample Gumbel noise: -log(-log(U))  ≡  log(exp(logit)) / (-log(U))^T
    noise = mx.random.uniform(shape=logits.shape)
    # Clamp to avoid log(0)
    noise = mx.clip(noise, 1e-10, 1.0)
    gumbel = (-mx.log(noise)) ** temperature
    return mx.exp(logits) / gumbel


# ---------------------------------------------------------------------------
# Step-size schedule
# ---------------------------------------------------------------------------


def get_num_transfer_tokens(
    mask_index: mx.array,
    steps: int,
) -> np.ndarray:
    """Compute how many masked tokens to unmask at each denoising step.

    Uses a linear (uniform) schedule: distributes the total number of masked
    tokens as evenly as possible across the given number of steps.

    Args:
        mask_index: Boolean array of shape (B, L); True where tokens are masked.
        steps: Number of denoising steps.

    Returns:
        Integer numpy array of shape (B, steps) giving per-step token counts.
    """
    mask_num = np.array(mask_index.sum(axis=1))  # (B,)
    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer = np.zeros((mask_num.shape[0], steps), dtype=np.int64) + base[:, None]
    for i, rem in enumerate(remainder):
        num_transfer[i, :rem] += 1

    return num_transfer


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------


def generate(
    model: Model,
    prompt: mx.array,
    attention_mask: mx.array | None = None,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    verbose: bool = False,
) -> mx.array:
    """Generate tokens using iterative masked diffusion.

    The algorithm:
      1. Append ``gen_length`` [MASK] tokens to the prompt.
      2. For each block (to support semi-autoregressive generation):
         a. Pre-compute how many tokens to unmask at each step.
         b. At each step:
            - Run the model to predict all token positions.
            - (Optional) apply classifier-free guidance.
            - Score each masked position by predicted confidence.
            - Commit the top-k most confident predictions; keep the rest masked.
      3. Return the completed sequence.

    Args:
        model: A trained ``Model`` (mask predictor).
        prompt: Token ids of shape (1, L_prompt) or (B, L_prompt).
        attention_mask: Optional padding mask for the prompt of shape (B, L_prompt).
        steps: Total denoising steps (≤ gen_length).  More steps → better quality.
        gen_length: Number of new tokens to generate.
        block_length: Generate this many tokens per semi-autoregressive block.
            Must divide gen_length exactly.  Set equal to gen_length for
            standard (non-semi-autoregressive) generation.
        temperature: Gumbel noise temperature; 0 = greedy.
        cfg_scale: Classifier-free guidance scale.  0 = disabled.
            When >0, runs an extra unconditional forward pass at each step.
        remasking: Strategy for deciding which tokens to re-mask between steps:
            - ``"low_confidence"`` (default): re-mask predictions with the
              lowest softmax probability (works best).
            - ``"random"``: re-mask a random subset.
        verbose: Print progress information.

    Returns:
        Completed token ids of shape (B, L_prompt + gen_length).
    """
    mask_token_id = model.args.mask_token_id
    B, L_prompt = prompt.shape

    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    assert steps % (gen_length // block_length) == 0, (
        "steps must be divisible by the number of blocks"
    )

    # Build the working sequence: [prompt | MASK … MASK]
    mask_fill = mx.full((B, gen_length), mask_token_id, dtype=mx.int32)
    x = mx.concatenate([prompt, mask_fill], axis=1)  # (B, L_prompt + gen_length)

    # Extend attention mask to cover the generation region
    if attention_mask is not None:
        gen_attn = mx.ones((B, gen_length), dtype=attention_mask.dtype)
        attention_mask = mx.concatenate([attention_mask, gen_attn], axis=1)

    # Remember which positions are part of the prompt (never unmask these)
    prompt_mask = x != mask_token_id  # (B, L_total)

    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    if verbose:
        print(f"Generating {gen_length} tokens in {num_blocks} block(s), "
              f"{steps_per_block} step(s) each …")

    for block_idx in range(num_blocks):
        block_start = L_prompt + block_idx * block_length
        block_end = L_prompt + (block_idx + 1) * block_length

        # Compute the per-step token-transfer schedule for this block
        block_mask_index = x[:, block_start:block_end] == mask_token_id
        mx.eval(block_mask_index)
        num_transfer = get_num_transfer_tokens(
            np.array(block_mask_index), steps_per_block
        )  # numpy (B, steps_per_block)

        for step in range(steps_per_block):
            # ---- forward pass ------------------------------------------------
            if cfg_scale > 0.0:
                # Classifier-free guidance: run conditional and unconditional
                # (unconditional = prompt tokens also replaced by [MASK])
                un_x = mx.where(prompt_mask, mask_token_id, x)
                x_cat = mx.concatenate([x, un_x], axis=0)  # (2B, L)

                if attention_mask is not None:
                    attn_cat = mx.concatenate([attention_mask, attention_mask], axis=0)
                    logits_cat = model(x_cat, attn_cat)
                else:
                    logits_cat = model(x_cat)

                logits_cond = logits_cat[:B]
                logits_uncond = logits_cat[B:]
                logits = logits_uncond + (cfg_scale + 1.0) * (
                    logits_cond - logits_uncond
                )
            else:
                logits = model(x, attention_mask)  # (B, L, V)

            # ---- sample predictions -----------------------------------------
            logits_noisy = add_gumbel_noise(logits, temperature)
            x0 = mx.argmax(logits_noisy, axis=-1)  # (B, L)

            # ---- confidence scoring -----------------------------------------
            mask_index = x == mask_token_id  # (B, L)

            if remasking == "low_confidence":
                p = mx.softmax(logits.astype(mx.float32), axis=-1)  # (B, L, V)
                # Gather probability of the predicted token at each position
                x0_p = p[
                    mx.arange(B)[:, None],
                    mx.arange(x.shape[1])[None, :],
                    x0,
                ]  # (B, L)
            elif remasking == "random":
                x0_p = mx.random.uniform(shape=(B, x.shape[1]))
            else:
                raise ValueError(f"Unknown remasking strategy: {remasking!r}")

            # Mask confidence scores for positions beyond the current block
            # and for already-unmasked (prompt / previously committed) positions
            beyond_block = mx.arange(x.shape[1])[None, :] >= block_end
            x0_p = mx.where(beyond_block | ~mask_index, float("-inf"), x0_p)

            # Best prediction for each position (use committed token if not masked)
            x0 = mx.where(mask_index, x0, x)

            # ---- commit top-k tokens ----------------------------------------
            # For each batch element, select the num_transfer[b, step] positions
            # with the highest confidence and commit them.
            #
            # Implementation: double-argsort gives rank (0 = most confident).
            # A position is transferred if its rank < num_to_unmask AND
            # it was masked.
            k_vals = mx.array(num_transfer[:, step])  # (B,)

            # argsort twice: first gives sorted indices, second gives rank
            sorted_idx = mx.argsort(-x0_p, axis=-1)   # (B, L) descending
            ranks = mx.argsort(sorted_idx, axis=-1)    # (B, L) rank of each pos

            transfer_mask = ranks < k_vals[:, None]    # (B, L)

            x = mx.where(transfer_mask, x0, x)
            mx.eval(x)

            if verbose and (step + 1) % max(1, steps_per_block // 4) == 0:
                n_remaining = int((x == mask_token_id).sum())
                print(f"  block {block_idx + 1}/{num_blocks}, "
                      f"step {step + 1}/{steps_per_block}: "
                      f"{n_remaining} mask tokens remaining")

    return x


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(model_path: str) -> tuple[Model, object]:
    """Load an MLX diffusion LM model and its tokenizer.

    The model directory should contain:
      - ``config.json``: ModelArgs fields (as saved by ``convert.py``).
      - ``weights.safetensors`` (or sharded ``weights-00001-of-XXXX.safetensors``).
      - A HuggingFace tokenizer (``tokenizer.json``, ``tokenizer_config.json``, …).

    Args:
        model_path: Path to the model directory.

    Returns:
        (model, tokenizer) tuple ready for inference.
    """
    import glob

    from transformers import AutoTokenizer

    path = Path(model_path)

    # Load config
    with open(path / "config.json") as f:
        cfg = json.load(f)
    args = ModelArgs.from_dict(cfg)
    model = Model(args)

    # Load weights (support sharding)
    weight_files = sorted(glob.glob(str(path / "*.safetensors")))
    if not weight_files:
        raise FileNotFoundError(f"No .safetensors files found in {path}")

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf).items())

    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    return model, tokenizer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate text with a diffusion LM")

    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to MLX model directory")
    parser.add_argument("--prompt", type=str,
                        default="The quick brown fox")
    parser.add_argument("--gen-length", type=int, default=64,
                        help="Number of tokens to generate")
    parser.add_argument("--steps", type=int, default=None,
                        help="Denoising steps (default: gen_length)")
    parser.add_argument("--block-length", type=int, default=None,
                        help="Semi-autoregressive block size (default: gen_length)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Gumbel noise temperature (0 = greedy)")
    parser.add_argument("--cfg-scale", type=float, default=0.0,
                        help="Classifier-free guidance scale (0 = disabled)")
    parser.add_argument("--remasking", type=str, default="low_confidence",
                        choices=["low_confidence", "random"])
    parser.add_argument("--chat", action="store_true",
                        help="Apply instruct chat template to the prompt")
    parser.add_argument("--demo", action="store_true",
                        help="Run a quick demo with a tiny random model (no weights needed)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    gen_length = args.gen_length
    steps = args.steps or gen_length
    block_length = args.block_length or gen_length

    if args.demo:
        # ---- tiny synthetic demo (no real weights) --------------------------
        print("Running demo with a tiny random model …\n")
        demo_args = ModelArgs(
            d_model=128, n_layers=2, n_heads=4, n_kv_heads=4,
            mlp_hidden_size=256, vocab_size=1001, mask_token_id=1000,
        )
        model = Model(demo_args)
        mx.eval(model.parameters())

        prompt_ids = mx.array([[1, 2, 3, 4, 5]])  # fake prompt
        t0 = time.perf_counter()
        out = generate(
            model, prompt_ids,
            steps=8, gen_length=16, block_length=16,
            temperature=1.0, verbose=args.verbose,
        )
        mx.eval(out)
        elapsed = time.perf_counter() - t0
        print(f"Output token ids: {out[0].tolist()}")
        print(f"Generated in {elapsed:.2f}s")
        return

    if args.model_path is None:
        parser.error("--model-path is required (or use --demo)")

    model, tokenizer = load_model(args.model_path)

    prompt_text = args.prompt
    if args.chat:
        messages = [{"role": "user", "content": prompt_text}]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    enc = tokenizer(
        [prompt_text],
        return_tensors="np",
        add_special_tokens=not args.chat,
        padding=True,
    )
    input_ids = mx.array(enc["input_ids"])
    attention_mask = mx.array(enc["attention_mask"])

    print(f"Prompt ({input_ids.shape[1]} tokens): {prompt_text!r}")
    print(f"Generating {gen_length} tokens with {steps} steps …\n")

    t0 = time.perf_counter()
    out = generate(
        model,
        input_ids,
        attention_mask=attention_mask,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        remasking=args.remasking,
        verbose=args.verbose,
    )
    mx.eval(out)
    elapsed = time.perf_counter() - t0

    response_ids = out[:, input_ids.shape[1]:]
    response = tokenizer.batch_decode(
        np.array(response_ids), skip_special_tokens=True
    )

    for i, text in enumerate(response):
        print(f"[{i}] {text}")
        print("-" * 60)

    print(f"\nGenerated {gen_length} tokens in {elapsed:.2f}s "
          f"({gen_length / elapsed:.1f} tok/s)")


if __name__ == "__main__":
    main()
