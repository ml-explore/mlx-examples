# Copyright © 2024 Apple Inc.

import argparse
import json
import math
import re
import types
from pathlib import Path

import mlx.optimizers as optim
import numpy as np
import yaml
from mlx.utils import tree_flatten

from .tuner.trainer import TrainingArgs, TrainingCallback, evaluate, train
from .tuner.utils import linear_to_lora_layers
from .utils import load

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
    "data": "data/",
    "seed": 0,
    "lora_layers": 16,
    "batch_size": 4,
    "iters": 1000,
    "val_batches": 25,
    "learning_rate": 1e-5,
    "steps_per_report": 10,
    "steps_per_eval": 200,
    "resume_adapter_file": None,
    "adapter_file": "adapters.npz",
    "save_every": 100,
    "test": False,
    "test_batches": 500,
    "max_seq_length": 2048,
    "lora_parameters": {"rank": 8, "alpha": 16, "dropout": 0.0, "scale": 10.0},
}


def build_parser():
    parser = argparse.ArgumentParser(description="LoRA or QLoRA finetuning.")
    parser.add_argument(
        "--model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        help="The maximum number of tokens to generate",
    )
    parser.add_argument("--temp", type=float, help="The sampling temperature")
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        help="The prompt for generation",
    )

    # Training args
    parser.add_argument(
        "--train",
        action="store_true",
        help="Do training",
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Directory with {train, valid, test}.jsonl files",
    )
    parser.add_argument(
        "--lora-layers",
        type=int,
        help="Number of layers to fine-tune",
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
        help="Load path to resume training with the given adapter weights.",
    )
    parser.add_argument(
        "--adapter-file",
        type=str,
        help="Save/load path for the trained adapter weights.",
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
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")
    return parser


class Dataset:
    """
    Light-weight wrapper to hold lines from a jsonl file
    """

    def __init__(self, path: Path, key: str = "text"):
        if not path.exists():
            self._data = None
        else:
            with open(path, "r") as fid:
                self._data = [json.loads(l) for l in fid]
        self._key = key

    def __getitem__(self, idx: int):
        return self._data[idx][self._key]

    def __len__(self):
        if self._data is None:
            return 0
        return len(self._data)


def load_dataset(args):
    names = ("train", "valid", "test")
    train, valid, test = (Dataset(Path(args.data) / f"{n}.jsonl") for n in names)
    if args.train and len(train) == 0:
        raise ValueError(
            "Training set not found or empty. Must provide training set for fine-tuning."
        )
    if args.train and len(valid) == 0:
        raise ValueError(
            "Validation set not found or empty. Must provide validation set for fine-tuning."
        )
    if args.test and len(test) == 0:
        raise ValueError(
            "Test set not found or empty. Must provide test set for evaluation."
        )
    return train, valid, test


def run(args, training_callback: TrainingCallback = None):
    np.random.seed(args.seed)

    print("Loading pretrained model")
    model, tokenizer = load(args.model)

    # Freeze all layers
    model.freeze()
    # Convert linear layers to lora layers and unfreeze in the process
    linear_to_lora_layers(model, args.lora_layers, args.lora_parameters)

    p = sum(v.size for _, v in tree_flatten(model.parameters())) / 10**6
    print(f"Total parameters {p:.3f}M")
    p = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
    print(f"Trainable parameters {p:.3f}M")

    print("Loading datasets")
    train_set, valid_set, test_set = load_dataset(args)

    # Resume training the given adapters.
    if args.resume_adapter_file is not None:
        print(f"Loading pretrained adapters from {args.resume_adapter_file}")
        model.load_weights(args.resume_adapter_file, strict=False)
    # init training args
    trainingArgs = TrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters,
        val_batches=args.val_batches,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        steps_per_save=args.save_every,
        adapter_file=args.adapter_file,
        max_seq_length=args.max_seq_length,
        grad_checkpoint=args.grad_checkpoint,
    )
    if args.train:
        print("Training")
        model.train()
        opt = optim.Adam(learning_rate=args.learning_rate)
        # Train model
        train(
            model=model,
            tokenizer=tokenizer,
            args=trainingArgs,
            optimizer=opt,
            train_dataset=train_set,
            val_dataset=valid_set,
            training_callback=training_callback,
        )

    # Load the LoRA adapter weights which we assume should exist by this point
    if not Path(args.adapter_file).is_file():
        raise ValueError(
            f"Adapter file {args.adapter_file} missing. "
            "Use --train to learn and save the adapters.npz."
        )
    model.load_weights(args.adapter_file, strict=False)

    if args.test:
        print("Testing")
        model.eval()

        test_loss = evaluate(
            model=model,
            dataset=test_set,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_batches=args.test_batches,
        )

        test_ppl = math.exp(test_loss)

        print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")

    if args.prompt is not None:
        raise NotImplementedError(
            "Please use mlx_lm.generate with trained adapter for generation."
        )


if __name__ == "__main__":
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
            if not args.get(k, None):
                args[k] = v

    # Update defaults for unspecified parameters
    for k, v in CONFIG_DEFAULTS.items():
        if not args.get(k, None):
            args[k] = v
    run(types.SimpleNamespace(**args))
