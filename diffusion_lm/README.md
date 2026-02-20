# Diffusion Language Model (LLaDA-style) for MLX

An MLX implementation of **masked diffusion language models**, following the approach from:

> **LLaDA: Large Language Diffusion with mAsking**
> Liao et al., 2025 — [arXiv:2502.09992](https://arxiv.org/abs/2502.09992)
> GitHub: [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA)

## What is a masked diffusion LM?

Instead of predicting the next token autoregressively, a masked diffusion LM
works bidirectionally:

- **Training (forward process)**: randomly mask a fraction of tokens in the input
  sequence with a special `[MASK]` token.  The fraction is sampled per-sequence
  from U(ε, 1).  The model (a bidirectional transformer) learns to predict the
  original token at every masked position simultaneously.
- **Inference (reverse process)**: start with a fully-masked response, then
  iteratively unmask the most-confident predictions until the sequence is
  complete.

The training objective is a weighted cross-entropy loss that is provably an
upper bound on the negative log-likelihood, making the model a proper generative
model.

## Files

| File | Description |
|---|---|
| `model.py` | Bidirectional transformer (mask predictor). LLaMA-style blocks without the causal mask. |
| `train.py` | Pre-training and supervised fine-tuning (SFT) training script. |
| `generate.py` | Iterative-unmasking generation with Gumbel-noise sampling, semi-autoregressive blocks, and classifier-free guidance. |
| `convert.py` | Convert a HuggingFace LLaDA checkpoint to MLX safetensors format. |

## Setup

```bash
pip install mlx transformers huggingface_hub
```

## Quick demo (no weights required)

```bash
python generate.py --demo
```

This runs a tiny 2-layer model with random weights to verify the generation
loop works end-to-end.

## Using a pre-trained LLaDA model

### Step 1 — Convert from HuggingFace

```bash
python convert.py \
    --hf-path GSAI-ML/LLaDA-8B-Instruct \
    --mlx-path mlx_llada_instruct
```

To save memory with 4-bit quantization:

```bash
python convert.py \
    --hf-path GSAI-ML/LLaDA-8B-Instruct \
    --mlx-path mlx_llada_instruct_4bit \
    -q
```

### Step 2 — Generate text

```bash
python generate.py \
    --model-path mlx_llada_instruct \
    --prompt "What is the capital of France?" \
    --gen-length 128 \
    --steps 128 \
    --chat
```

Useful flags:

| Flag | Default | Description |
|---|---|---|
| `--gen-length` | 64 | Tokens to generate |
| `--steps` | gen_length | Denoising steps (more = better quality, slower) |
| `--block-length` | gen_length | Block size for semi-autoregressive generation |
| `--temperature` | 0.0 | Gumbel noise temperature (0 = greedy) |
| `--cfg-scale` | 0.0 | Classifier-free guidance strength |
| `--remasking` | low_confidence | Remasking strategy: `low_confidence` or `random` |
| `--chat` | False | Apply the instruct chat template |

## Training from scratch

### Pre-training

Prepare a JSONL file where each line is `{"text": "..."}`:

```bash
python train.py \
    --data data/train.jsonl \
    --tokenizer meta-llama/Meta-Llama-3-8B \
    --d_model 512 --n_layers 8 --n_heads 8 --mlp_hidden_size 2048 \
    --batch_size 16 --iters 50000 --lr 3e-4
```

### Supervised fine-tuning (SFT)

Prepare a JSONL file where each line is `{"prompt": "...", "response": "..."}`:

```bash
python train.py \
    --sft \
    --data data/sft.jsonl \
    --tokenizer meta-llama/Meta-Llama-3-8B \
    --d_model 512 --n_layers 8 --n_heads 8 --mlp_hidden_size 2048 \
    --batch_size 8 --iters 10000 --lr 1e-4
```

For a quick smoke-test with a tiny model on random data:

```bash
python train.py --d_model 128 --n_layers 2 --n_heads 4 --mlp_hidden_size 256 \
    --batch_size 4 --iters 100 --log_every 10
```

## Architecture

The model is a standard transformer with one key change: **no causal mask** in
self-attention.  Every token can attend to every other token, making the model
bidirectional.

| Component | Detail |
|---|---|
| Attention | Multi-head, bidirectional (no causal mask), RoPE position encoding |
| FFN | SwiGLU: `silu(gate_proj(x)) * up_proj(x)` → `down_proj` |
| Normalisation | RMSNorm (pre-norm, before attention and FFN) |
| Special token | `[MASK]` token replaces corrupted positions during both training and inference |

LLaDA-8B uses the same hyper-parameters as LLaMA-3-8B (d_model=4096, 32 layers,
32 heads, mlp_hidden_size=14336, rope_theta=500000) with the LLaMA-3 tokenizer
(vocab_size=126464, mask_token_id=126336).

## Generation algorithm

```
Input:  prompt tokens  [p₁ p₂ … pₙ]
Init:   x = [p₁ … pₙ | M M … M]   (M = [MASK], gen_length masks)

For each block b in 1..num_blocks:
    Compute token schedule:  how many tokens to unmask at each step
    For step t in 1..steps_per_block:
        logits ← model(x)                 # bidirectional forward pass
        x₀     ← argmax(logits + Gumbel)  # sample predictions
        conf   ← softmax_prob(x₀)         # confidence scores
        k      ← num_transfer[b, t]       # tokens to commit this step
        Commit the k highest-confidence masked positions in x
Return x[n+1:]
```

The **linear noise schedule** distributes unmaskings evenly: if there are `N`
masked tokens and `T` steps, approximately `N/T` tokens are committed per step.

## References

- Liao et al. (2025). *LLaDA: Large Language Diffusion with mAsking*. arXiv:2502.09992.
- Austin et al. (2021). *Structured Denoising Diffusion Models in Discrete State-Spaces*.
- Shi et al. (2024). *Simplified and Generalized Masked Diffusion for Discrete Data*. arXiv:2406.04329.
