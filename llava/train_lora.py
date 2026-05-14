# Copyright © 2024 Apple Inc.

import argparse
import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import requests
from mlx.utils import tree_flatten
from PIL import Image
from transformers import AutoProcessor

from llava import LlavaModel


class LoRALinear(nn.Module):
    @staticmethod
    def from_linear(linear: nn.Linear, rank: int = 8, scale: float = 20.0):
        output_dims, input_dims = linear.weight.shape
        lora_linear = LoRALinear(input_dims, output_dims, rank, scale)
        lora_linear.linear = linear
        return lora_linear

    def __init__(self, input_dims: int, output_dims: int, rank: int, scale: float):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=False)
        self.scale = scale
        bound = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-bound,
            high=bound,
            shape=(input_dims, rank),
        )
        self.lora_b = mx.zeros(shape=(rank, output_dims))

    def __call__(self, x):
        dtype = self.linear.weight.dtype
        y = self.linear(x.astype(dtype))
        z = (x @ self.lora_a) @ self.lora_b
        return y + self.scale * z


class Dataset:
    def __init__(self, path):
        self._data = []
        with open(path, "r") as fid:
            for line in fid:
                self._data.append(json.loads(line))

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


def build_parser():
    parser = argparse.ArgumentParser(description="Fine-tune LLaVA with LoRA.")
    parser.add_argument(
        "--model",
        type=str,
        default="llava-hf/llava-v1.6-mistral-7b-hf",
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="JSONL file with image, prompt, and text fields.",
    )
    parser.add_argument(
        "--valid-data",
        type=str,
        default=None,
        help="Optional JSONL validation file.",
    )
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-layers", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-scale", type=float, default=20.0)
    parser.add_argument("--steps-per-report", type=int, default=10)
    parser.add_argument("--steps-per-eval", type=int, default=200)
    parser.add_argument("--adapter-file", type=str, default="llava_adapters.npz")
    parser.add_argument("--resume-adapter-file", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def load_image(image_source):
    if image_source.startswith(("http://", "https://")):
        response = requests.get(image_source, stream=True)
        response.raise_for_status()
        return Image.open(response.raw).convert("RGB")
    if Path(image_source).is_file():
        return Image.open(image_source).convert("RGB")
    raise ValueError(f"The image {image_source} must be a valid URL or file path.")


def prepare_example(processor, example):
    image = load_image(example["image"])
    prompt = example["prompt"]
    text = example["text"]

    full_inputs = processor(
        images=image,
        text=prompt + text,
        return_tensors="np",
    )
    prompt_inputs = processor(
        images=image,
        text=prompt,
        return_tensors="np",
    )

    input_ids = full_inputs["input_ids"]
    prompt_length = prompt_inputs["input_ids"].shape[1]
    loss_mask = np.zeros((1, input_ids.shape[1] - 1), dtype=np.float32)
    loss_mask[:, max(prompt_length - 1, 0) :] = 1

    image_sizes = full_inputs.get("image_sizes")
    image_sizes = mx.array(image_sizes) if image_sizes is not None else None

    return (
        mx.array(input_ids[:, :-1]),
        mx.array(input_ids[:, 1:]),
        mx.array(full_inputs["pixel_values"]),
        image_sizes,
        mx.array(loss_mask),
    )


def iterate_batches(dataset, processor, train=False):
    while True:
        indices = np.arange(len(dataset))
        if train:
            indices = np.random.permutation(indices)
        for idx in indices:
            yield prepare_example(processor, dataset[idx])
        if not train:
            break


def loss(model, input_ids, targets, pixel_values, image_sizes, loss_mask):
    logits, _ = model(input_ids, pixel_values, image_sizes=image_sizes)
    logits = logits.astype(mx.float32)
    ce = nn.losses.cross_entropy(logits, targets) * loss_mask
    ntoks = loss_mask.sum()
    return ce.sum() / ntoks, ntoks


def evaluate(model, dataset, processor, num_batches=25):
    losses = []
    ntokens = 0
    for _, batch in zip(range(num_batches), iterate_batches(dataset, processor)):
        batch_loss, toks = loss(model, *batch)
        losses.append((batch_loss * toks).item())
        ntokens += toks.item()
    return np.sum(losses) / ntokens


def apply_lora(model, num_layers, rank, scale):
    model.vision_tower.freeze()
    model.multi_modal_projector.freeze()
    model.language_model.freeze()

    layers = model.language_model.model.layers[-num_layers:]
    for layer in layers:
        layer.self_attn.q_proj = LoRALinear.from_linear(
            layer.self_attn.q_proj, rank=rank, scale=scale
        )
        layer.self_attn.v_proj = LoRALinear.from_linear(
            layer.self_attn.v_proj, rank=rank, scale=scale
        )


def train(model, train_set, valid_set, processor, optimizer, args):
    loss_value_and_grad = nn.value_and_grad(model, loss)
    losses = []

    for it, batch in zip(
        range(args.iters),
        iterate_batches(train_set, processor, train=True),
    ):
        (loss_value, _), grad = loss_value_and_grad(model, *batch)
        optimizer.update(model, grad)
        mx.eval(model.parameters(), optimizer.state, loss_value)

        losses.append(loss_value.item())

        if (it + 1) % args.steps_per_report == 0:
            print(f"Iter {it + 1}: Train loss {np.mean(losses):.3f}")
            losses = []

        if valid_set is not None and (it == 0 or (it + 1) % args.steps_per_eval == 0):
            val_loss = evaluate(model, valid_set, processor)
            print(f"Iter {it + 1}: Val loss {val_loss:.3f}")

        if (it + 1) % args.save_every == 0:
            mx.savez(
                args.adapter_file, **dict(tree_flatten(model.trainable_parameters()))
            )
            print(f"Iter {it + 1}: Saved adapter weights to {args.adapter_file}.")


def main():
    args = build_parser().parse_args()
    if args.batch_size != 1:
        raise ValueError("LLaVA LoRA fine-tuning currently supports --batch-size 1.")
    np.random.seed(args.seed)

    print("Loading model")
    processor = AutoProcessor.from_pretrained(args.model, use_fast=False)
    model = LlavaModel.from_pretrained(args.model)
    apply_lora(model, args.lora_layers, args.lora_rank, args.lora_scale)

    if args.resume_adapter_file is not None:
        print(f"Loading adapter weights from {args.resume_adapter_file}")
        model.load_weights(args.resume_adapter_file, strict=False)

    train_set = Dataset(args.train_data)
    valid_set = Dataset(args.valid_data) if args.valid_data is not None else None

    total_params = sum(v.size for _, v in tree_flatten(model.parameters())) / 10**6
    trainable_params = (
        sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
    )
    print(f"Total parameters {total_params:.3f}M")
    print(f"Trainable parameters {trainable_params:.3f}M")

    optimizer = optim.Adam(learning_rate=args.learning_rate)
    train(model, train_set, valid_set, processor, optimizer, args)

    mx.savez(args.adapter_file, **dict(tree_flatten(model.trainable_parameters())))
    print(f"Saved adapter weights to {args.adapter_file}.")


if __name__ == "__main__":
    main()
