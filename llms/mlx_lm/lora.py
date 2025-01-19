# Copyright © 2024 Apple Inc.

import argparse
import math
import os
import re
import types
from pathlib import Path

import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yaml

from .tokenizer_utils import TokenizerWrapper
from .tuner.datasets import load_dataset
from .tuner.trainer import TrainingArgs, TrainingCallback, evaluate, train
from .tuner.dpo_trainer import DPOTrainingArgs, evaluate_dpo, train_dpo
from .tuner.orpo_trainer import ORPOTrainingArgs, evaluate_orpo, train_orpo
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
    "training_mode": "normal",
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
    "config": None,
    "grad_checkpoint": False,
    "lr_schedule": None,
    "lora_parameters": {"rank": 8, "alpha": 16, "dropout": 0.0, "scale": 10.0},
    "beta": 0.1,
    "dpo_loss_type": "sigmoid",
    "is_reference_free": False,
    "delta": 50.0,
    "reference_model_path": None,
    "reward_scaling": 1.0,
}


def build_parser():
    parser = argparse.ArgumentParser(description="LoRA or QLoRA finetuning.")
    parser.add_argument(
        "--model",
        type=str,
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
        help="Type of fine-tuning to perform: lora, dora, or full.",
    )
    parser.add_argument(
        "--training-mode",
        type=str,
        choices=["normal", "dpo", "orpo"],
        help="Training mode: normal, DPO or ORPO.",
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
        help="Evaluate on the test set after training.",
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
        type=str,
        help="A YAML configuration file with the training options.",
    )
    parser.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="Use gradient checkpointing to reduce memory use.",
        default=None,
    )
    parser.add_argument("--beta", type=float)
    parser.add_argument("--dpo-loss-type", type=str, choices=["sigmoid", "hinge", "ipo", "dpop"])
    parser.add_argument("--is-reference-free", action="store_true")
    parser.add_argument("--delta", type=float)
    parser.add_argument("--reference-model-path", type=str)
    parser.add_argument("--reward-scaling", type=float, help="Scaling factor for offline rewards.")
    parser.add_argument("--seed", type=int, help="The PRNG seed.")
    return parser


def train_model(
    args,
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    train_set,
    valid_set,
    training_callback: TrainingCallback = None,
):
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

    model.train()
    opt = optim.Adam(
        learning_rate=(
            build_schedule(args.lr_schedule) if args.lr_schedule else args.learning_rate
        )
    )
    
    # Train model based on training mode
    if args.training_mode == "dpo":
        training_args = DPOTrainingArgs(
            batch_size=args.batch_size,
            iters=args.iters,
            val_batches=args.val_batches,
            steps_per_report=args.steps_per_report,
            steps_per_eval=args.steps_per_eval,
            steps_per_save=args.save_every,
            adapter_file=adapter_file,
            max_seq_length=args.max_seq_length,
            grad_checkpoint=args.grad_checkpoint,
            beta=args.beta,
            loss_type=args.dpo_loss_type,
            is_reference_free=args.is_reference_free,
            delta=args.delta,
            reference_model_path=args.reference_model_path,
        )
        
        if args.reference_model_path:
            reference_model, _ = load(args.reference_model_path)
        else:
            reference_model, _ = load(args.model)
            
        train_dpo(
            model=model,
            reference_model=reference_model.freeze(),
            tokenizer=tokenizer,
            optimizer=opt,
            train_dataset=train_set,
            val_dataset=valid_set,
            args=training_args,
            training_callback=training_callback,
        )
    elif args.training_mode == "orpo":
        training_args = ORPOTrainingArgs(
            batch_size=args.batch_size,
            iters=args.iters,
            val_batches=args.val_batches,
            steps_per_report=args.steps_per_report,
            steps_per_eval=args.steps_per_eval,
            steps_per_save=args.save_every,
            adapter_file=adapter_file,
            max_seq_length=args.max_seq_length,
            grad_checkpoint=args.grad_checkpoint,
            beta=args.beta,
            reward_scaling=args.reward_scaling,
        )
            
        train_orpo(
            model=model,
            tokenizer=tokenizer,
            optimizer=opt,
            train_dataset=train_set,
            val_dataset=valid_set,
            args=training_args,
            training_callback=training_callback,
        )
    else:
        training_args = TrainingArgs(
            batch_size=args.batch_size,
            iters=args.iters,
            val_batches=args.val_batches,
            steps_per_report=args.steps_per_report,
            steps_per_eval=args.steps_per_eval,
            steps_per_save=args.save_every,
            adapter_file=adapter_file,
            max_seq_length=args.max_seq_length,
            grad_checkpoint=args.grad_checkpoint
        )

        train(
            model=model,
            tokenizer=tokenizer,
            optimizer=opt,
            train_dataset=train_set,
            val_dataset=valid_set,
            args=training_args,
            training_callback=training_callback,
        )


def evaluate_model(args, model: nn.Module, tokenizer: TokenizerWrapper, test_set):
    model.eval()

    if args.training_mode == "dpo":
        if args.reference_model_path:
            reference_model, _ = load(args.reference_model_path)
        else:
            reference_model = model

        test_loss, test_rewards = evaluate_dpo(
            model=model,
            reference_model=reference_model,
            dataset=test_set,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_batches=args.test_batches,
            max_seq_length=args.max_seq_length,
            beta=args.beta,
            delta=args.delta,
            loss_type=args.dpo_loss_type,
        )
        print(f"Test loss {test_loss:.3f}, Rewards: {test_rewards[0]:.3f}, {test_rewards[1]:.3f}")
    elif args.training_mode == "orpo":
        test_loss, test_rewards = evaluate_orpo(
            model=model,
            dataset=test_set,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_batches=args.test_batches,
            max_seq_length=args.max_seq_length,
            beta=args.beta,
            reward_scaling=args.reward_scaling,
        )
        print(f"Test loss {test_loss:.3f}, Rewards: {test_rewards[0]:.3f}, {test_rewards[1]:.3f}")
    else:
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
        print(f"Training in {args.training_mode} mode")
        train_model(args, model, tokenizer, train_set, valid_set, training_callback)
    else:
        raise ValueError("Must provide at least one of --train or --test")

    if args.test:
        print("Testing")
        evaluate_model(args, model, tokenizer, test_set)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
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
