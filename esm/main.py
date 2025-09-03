import argparse

import mlx.core as mx

from esm import ESM2


def main():
    parser = argparse.ArgumentParser(description="ESM-2 MLX Inference")
    parser.add_argument(
        "--model-path",
        default="checkpoints/mlx-esm2_t33_650M_UR50D",
        help="Path to MLX model checkpoint",
    )
    parser.add_argument(
        "--sequence",
        default="MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
        help="Protein sequence to test (default: human insulin)",
    )
    parser.add_argument(
        "--mask-position",
        type=int,
        default=None,
        help="Position to mask (default: middle of sequence)",
    )
    args = parser.parse_args()

    # Load pretrained ESM-2 model and tokenizer
    tokenizer, model = ESM2.from_pretrained(args.model_path)

    # Determine sequence and position to mask
    sequence = args.sequence.upper()
    mask_pos = (
        args.mask_position if args.mask_position is not None else len(sequence) // 2
    )
    if mask_pos >= len(sequence):
        mask_pos = len(sequence) - 1
    original_aa = sequence[mask_pos]  # The original amino acid at masked position

    # Display input info
    print(f"Original sequence: {sequence}")
    print(f"Masked sequence: {sequence[:mask_pos]}<mask>{sequence[mask_pos+1:]}")
    print(f"Predicting position {mask_pos}: {original_aa}\n")

    # Tokenize sequence before and after the mask
    before = tokenizer.encode(sequence[:mask_pos], add_special_tokens=False)
    after = tokenizer.encode(sequence[mask_pos + 1 :], add_special_tokens=False)

    # Build token sequence with <cls>, <mask>, and <eos>
    tokens = mx.array(
        [
            [tokenizer.cls_id]
            + before.tolist()
            + [tokenizer.mask_id]
            + after.tolist()
            + [tokenizer.eos_id]
        ]
    )
    mask_token_pos = 1 + len(before)  # Position of <mask> token

    # Run model to get logits for each token position
    logits = model(tokens)["logits"]
    probs = mx.softmax(
        logits[0, mask_token_pos, :]
    )  # Softmax over vocabulary at mask position

    # Get top-5 most likely tokens
    top_indices = mx.argsort(probs)[-5:][::-1]

    # Print predictions
    print("Top predictions:")
    for i, idx in enumerate(top_indices):
        token = tokenizer.vocab[int(idx)]
        if token in tokenizer.vocab:
            prob = float(probs[idx])
            marker = "✓" if token == original_aa else " "
            print(f"{marker} {i+1}. {token}: {prob:.3f} ({prob*100:.1f}%)")


if __name__ == "__main__":
    main()
