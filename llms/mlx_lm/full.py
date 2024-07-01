import argparse
import json
import math
import re
import types
import os
from pathlib import Path

import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yaml
from mlx.utils import tree_flatten
from mlx.core import load as core_load
from mlx.core import save_safetensors as core_save_safetensors
from mlx.core import save_gguf as core_save_gguf

from .tuner.datasets import load_dataset
from .tuner.trainer import TrainingArgs, TrainingCallback, evaluate, train
from .tuner.utils import build_schedule
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
    "batch_size": 4,
    "iters": 1000,
    "val_batches": 25,
    "learning_rate": 1e-5,
    "steps_per_report": 10,
    "steps_per_eval": 200,
    "model_file": "model.safetensors",
    "save_every": 100,
    "test": False,
    "test_batches": 500,
    "max_seq_length": 2048,
    "lr_schedule": None,
}


def build_parser():
    parser = argparse.ArgumentParser(description="Full fine-tuning.")
    parser.add_argument(
        "--model",
        help="The path to the local model directory or Hugging Face repo.",
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
        "--model-file",
        type=str,
        help="Save/load path for the trained model weights.",
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


def run(args, training_callback: TrainingCallback = None):
    np.random.seed(args.seed)

    print("Loading pretrained model")
    model, tokenizer = load(args.model)
    
    # Load the model weights if they exist: To support Resuming from last file
    if Path(args.model_file).is_file():
        model.update(core_load(args.model_file))
        print(f"model file loaded: {args.model_file}")

    print("Loading datasets")
    train_set, valid_set, test_set = load_dataset(args, tokenizer)

    if args.train:
        print("Training")
        # init training args
        training_args = TrainingArgs(
            batch_size=args.batch_size,
            iters=args.iters,
            val_batches=args.val_batches,
            steps_per_report=args.steps_per_report,
            steps_per_eval=args.steps_per_eval,
            steps_per_save=args.save_every,
            max_seq_length=args.max_seq_length,
            grad_checkpoint=args.grad_checkpoint,
        )

        model.train()
        opt = optim.Adam(
            learning_rate=(
                build_schedule(args.lr_schedule)
                if args.lr_schedule
                else args.learning_rate
            )
        )
        # Train model
        train(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            optimizer=opt,
            train_dataset=train_set,
            val_dataset=valid_set,
            training_callback=training_callback,
        )

        # Save the model weights
        # ISSUE: in trainer.py, save_adapter saved already full size model. so this code is renaming only
        if training_args.adapter_file.rsplit('.',1)[-1] == args.model_file.rsplit('.',1)[-1]: #maybe npz?
          os.rename(training_args.adapter_file,args.model_file)
        else:
          model_tmp = core_load(training_args.adapter_file)
          if args.model_file.rsplit('.',1)[-1] == 'safetensors':
              core_save_safetensors(args.model_file, model_tmp)
          elif args.model_file.rsplit('.',1)[-1] == 'gguf':
              core_save_gguf(args.model_file, model_tmp)
          else:
              print(f"ERROR: only npz, gguf, safetensors format are available. use {training_args.adapter_file} file instead of {args.model_file}.")
          
        print(f"Saved final model weights to {args.model_file}.")

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
