# Copyright © 2025 Apple Inc.

"""
CLI-based inference script for MNIST.

This script loads a trained MNIST model and provides:
1. Random sample predictions with confidence scores
2. ASCII art visualization of digits (no matplotlib required)
3. Interactive mode to test specific samples

Usage:
    # First, train the model
    python main.py

    # Then run inference
    python infer.py

    # Or with custom model path
    python infer.py --model model.safetensors
"""

import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import mnist
from main import MLP


def load_model(model_path: str):
    """Load a trained MNIST model from safetensors file."""
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Please run 'python main.py' first to train a model."
        )

    # Create model with same architecture as main.py
    model = MLP(num_layers=2, input_dim=784, hidden_dim=32, output_dim=10)
    model.load_weights(model_path)
    mx.eval(model.parameters())

    return model


def show_mnist_digit(image, label=None):
    """Visualize MNIST image as ASCII art (28x28)."""
    image_2d = np.array(image.reshape(28, 28))

    # Normalize to 0-1 range
    if image_2d.max() > image_2d.min():
        image_2d = (image_2d - image_2d.min()) / (image_2d.max() - image_2d.min())

    # Convert to ASCII
    chars = " ·:-=+*#%@"
    result = []
    for row in image_2d:
        line = ""
        for pixel in row:
            char_idx = min(int(pixel * (len(chars) - 1)), len(chars) - 1)
            line += chars[char_idx] * 2  # Double width for square appearance
        result.append(line)

    if label is not None:
        print(f"\nTrue Label: {label}")
    print("\n".join(result))


def predict_samples(model, images, labels, num_samples=5):
    """Show predictions for random samples."""
    print("\n" + "=" * 60)
    print("Random Sample Predictions")
    print("=" * 60)

    indices = np.random.choice(len(labels), num_samples, replace=False)

    correct = 0
    for idx in indices:
        idx = int(idx)
        image = images[idx:idx+1]
        true_label = int(labels[idx].item())

        # Predict
        logits = model(image)
        predicted_label = int(mx.argmax(logits, axis=1).item())
        probabilities = mx.softmax(logits, axis=-1)[0]
        confidence = float(probabilities[predicted_label].item())

        # Check if correct
        is_correct = predicted_label == true_label
        if is_correct:
            correct += 1

        # Display result
        status = "✓" if is_correct else "✗"
        print(f"\n{status} Sample {idx}:")
        print(f"  True:       {true_label}")
        print(f"  Predicted:  {predicted_label}")
        print(f"  Confidence: {confidence*100:.1f}%")

        # Top 3 predictions
        top3_indices = mx.argsort(probabilities)[-3:][::-1]
        print(f"  Top 3:")
        for i in top3_indices:
            i_val = int(i.item())
            prob = float(probabilities[i_val].item())
            print(f"    {i_val}: {prob*100:.1f}%")

    print(f"\nAccuracy: {correct}/{num_samples} ({correct/num_samples*100:.0f}%)")


def interactive_mode(model, images, labels):
    """Interactive prediction mode."""
    print("\n" + "=" * 60)
    print("Interactive Mode")
    print(f"Enter an index (0-{len(labels)-1}) to visualize and predict")
    print("Enter 'r' for random sample, 'q' to quit")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nIndex: ").strip().lower()

            if user_input == 'q':
                print("Exiting...")
                break

            if user_input == 'r':
                idx = int(np.random.randint(0, len(labels)))
            else:
                idx = int(user_input)

            if idx < 0 or idx >= len(labels):
                print(f"Please enter a number between 0 and {len(labels)-1}")
                continue

            # Show image
            image = images[idx]
            true_label = int(labels[idx].item())
            show_mnist_digit(image, true_label)

            # Predict
            logits = model(image.reshape(1, -1))
            predicted_label = int(mx.argmax(logits, axis=1).item())
            probabilities = mx.softmax(logits, axis=-1)[0]
            confidence = float(probabilities[predicted_label].item())

            print(f"\nPrediction: {predicted_label} (Confidence: {confidence*100:.1f}%)")
            if predicted_label == true_label:
                print("✓ Correct!")
            else:
                print("✗ Wrong!")

        except ValueError:
            print("Invalid input. Please enter a number, 'r', or 'q'.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break


def main(args):
    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model)
    print("Model loaded successfully!")

    # Load test data
    print("Loading MNIST test data...")
    _, _, test_images, test_labels = map(mx.array, getattr(mnist, args.dataset)())
    print(f"Loaded {len(test_labels)} test samples")

    # Show random predictions
    if args.num_samples > 0:
        predict_samples(model, test_images, test_labels, args.num_samples)

    # Interactive mode
    if args.interactive:
        interactive_mode(model, test_images, test_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on trained MNIST model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model.safetensors",
        help="Path to trained model (default: model.safetensors)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist"],
        help="The dataset to use (default: mnist)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of random samples to predict (default: 5, 0 to skip)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive prediction mode",
    )
    args = parser.parse_args()

    main(args)
