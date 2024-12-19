# Copyright © 2024 Apple Inc.

import argparse
import math
import re
import types
from pathlib import Path

import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yaml

from .tokenizer_utils import TokenizerWrapper, no_bos_or_eos
from .tuner.datasets import load_dataset
from .tuner.trainer import TrainingArgs, TrainingCallback, evaluate, train
from .tuner.utils import (
    build_schedule,
    linear_to_lora_layers,
    load_adapters,
    print_trainable_parameters,
)
from .utils import load, save_config

yaml_loader = yaml.SafeLoader
yaml_loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        """^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)

CONFIG_DEFAULTS = {
    "model": "mlx_model",
    "train": False,
    "fine_tune_type": "lora",
    "data": "data/",
    "seed": 0,
    "num_layers": 16,
    "batch_size": 4,
    "iters": 1000,
    "val_batches": 25,
    "learning_rate": 1e-5,
    "steps_per_report": 10,
    "steps_per_eval": 200,
    "resume_adapter_file": None,
    "adapter_path": "adapters",
    "save_every": 100,
    "test": False,
    "test_batches": 500,
    "max_seq_length": 2048,
    "lr_schedule": None,
    "hf_datasets": None,
    "lora_parameters": {"rank": 8, "alpha": 16, "dropout": 0.0, "scale": 10.0},
    "response_template": None,
}


def build_parser():
    parser = argparse.ArgumentParser(description="LoRA or QLoRA finetuning.")
    parser.add_argument(
        "--model",
        help="The path to the local model directory or Hugging Face repo.",
    )

    # Training args
    parser.add_argument(
        "--train",
        action="store_true",
        help="Do training",
        default=None,
    )
    parser.add_argument(
        "--data",
        type=str,
        help=(
            "Directory with {train, valid, test}.jsonl files or the name "
            "of a Hugging Face dataset (e.g., 'mlx-community/wikisql')"
        ),
    )
    parser.add_argument(
        "--fine-tune-type",
        type=str,
        choices=["lora", "dora", "full"],
        default="lora",
        help="Type of fine-tuning to perform: lora, dora, or full.",
    )

    parser.add_argument(
        "--mask-inputs",
        dest="mask_inputs",
        action="store_true",
        help="Whether to mask the inputs when training. Default is False.",
        default=False,
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        help="Number of layers to fine-tune. Default is 16, use -1 for all.",
    )
    parser.add_argument("--batch-size", type=int, help="Minibatch size.")
    parser.add_argument("--iters", type=int, help="Iterations to train for.")
    parser.add_argument(
        "--val-batches",
        type=int,
        help="Number of validation batches, -1 uses the entire validation set.",
    )
    parser.add_argument("--learning-rate", type=float, help="Adam learning rate.")
    parser.add_argument(
        "--steps-per-report",
        type=int,
        help="Number of training steps between loss reporting.",
    )
    parser.add_argument(
        "--steps-per-eval",
        type=int,
        help="Number of training steps between validations.",
    )
    parser.add_argument(
        "--resume-adapter-file",
        type=str,
        help="Load path to resume training from the given fine-tuned weights.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Save/load path for the fine-tuned weights.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        help="Save the model every N iterations.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate on the test set after training",
        default=None,
    )
    parser.add_argument(
        "--test-batches",
        type=int,
        help="Number of test set batches, -1 uses the entire test set.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="A YAML configuration file with the training options",
    )
    parser.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="Use gradient checkpointing to reduce memory use.",
        default=None,
    )
    parser.add_argument("--seed", type=int, default=None, help="The PRNG seed")
    return parser


def train_model(
    args,
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    train_set,
    valid_set,
    training_callback: TrainingCallback = None,
):
    from .tuner.trainer import (
        default_loss,
        input_masked_loss,
        iterate_batches,
        iterate_completion_batches,
    )

    model.freeze()
    if args.fine_tune_type == "full":
        for l in model.layers[-min(args.num_layers, 0) :]:
            l.unfreeze()
    elif args.fine_tune_type in ["lora", "dora"]:
        # Convert linear layers to lora/dora layers and unfreeze in the process
        linear_to_lora_layers(
            model,
            args.num_layers,
            args.lora_parameters,
            use_dora=(args.fine_tune_type == "dora"),
        )
    else:
        raise ValueError(f"Received unknown fine-tune-type {args.fine_tune_type}")

    # Resume from weights if provided
    if args.resume_adapter_file is not None:
        print(f"Loading fine-tuned weights from {args.resume_adapter_file}")
        model.load_weights(args.resume_adapter_file, strict=False)

    print_trainable_parameters(model)

    adapter_path = Path(args.adapter_path)
    adapter_path.mkdir(parents=True, exist_ok=True)

    adapter_file = adapter_path / "adapters.safetensors"
    save_config(vars(args), adapter_path / "adapter_config.json")

    if isinstance(args.response_template, str):
        response_generation_tokens = tokenizer.encode(
            args.response_template, add_special_tokens=False
        )
    else:
        if not all([item.isinstance(int) for item in args.response_template]):
            raise ValueError(
                "Response template must be a list of integers if it is not a string."
            )
        response_generation_tokens = args.response_template

    # init training args
    training_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters,
        val_batches=args.val_batches,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        steps_per_save=args.save_every,
        adapter_file=adapter_file,
        max_seq_length=args.max_seq_length,
        grad_checkpoint=args.grad_checkpoint,
        response_generation_tokens=no_bos_or_eos(
            response_generation_tokens, tokenizer.bos_token_id, tokenizer.eos_token_id
        ),
    )

    model.train()
    opt = optim.Adam(
        learning_rate=(
            build_schedule(args.lr_schedule) if args.lr_schedule else args.learning_rate
        )
    )

    if args.mask_inputs:
        print("Masking inputs..")

    # Train model
    train(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        optimizer=opt,
        train_dataset=train_set,
        val_dataset=valid_set,
        training_callback=training_callback,
        iterate_batches=(
            iterate_completion_batches if args.mask_inputs else iterate_batches
        ),
        loss=input_masked_loss if args.mask_inputs else default_loss,
    )


def evaluate_model(args, model: nn.Module, tokenizer: TokenizerWrapper, test_set):
    model.eval()

    test_loss = evaluate(
        model=model,
        dataset=test_set,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_batches=args.test_batches,
        max_seq_length=args.max_seq_length,
    )

    test_ppl = math.exp(test_loss)

    print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")


def run(args, training_callback: TrainingCallback = None):
    np.random.seed(args.seed)

    print("Loading pretrained model")
    model, tokenizer = load(args.model)

    print("Loading datasets")
    train_set, valid_set, test_set = load_dataset(args, tokenizer)

    if args.test and not args.train:
        # Allow testing without LoRA layers by providing empty path
        if args.adapter_path != "":
            load_adapters(model, args.adapter_path)

    elif args.train:
        print("Training")
        train_model(args, model, tokenizer, train_set, valid_set, training_callback)
    else:
        raise ValueError("Must provide at least one of --train or --test")

    if args.test:
        print("Testing")
        evaluate_model(args, model, tokenizer, test_set)


def main():
    parser = build_parser()
    args = parser.parse_args()
    config = args.config
    args = vars(args)
    if config:
        print("Loading configuration file", config)
        with open(config, "r") as file:
            config = yaml.load(file, yaml_loader)
        # Prefer parameters from command-line arguments
        for k, v in config.items():
            if args.get(k, None) is None:
                args[k] = v

    # Update defaults for unspecified parameters
    for k, v in CONFIG_DEFAULTS.items():
        if args.get(k, None) is None:
            args[k] = v
    run(types.SimpleNamespace(**args))


if __name__ == "__main__":
    main()
