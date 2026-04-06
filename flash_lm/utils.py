import dataclasses
import importlib
import inspect
import json
import os
import re
import types
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import wandb

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import transformers
import yaml
from mlx.utils import tree_flatten

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


def load_config(config_path):
    config_path = Path(config_path)
    if config_path.suffix != ".yaml":
        config_path = config_path / "config.yaml"
    with open(config_path, "r") as fid:
        config = yaml.load(fid, yaml_loader)
        return types.SimpleNamespace(**config)


def save_config(dirname, config):
    with open(Path(dirname) / "config.yaml", "w") as fid:
        yaml.safe_dump(config.__dict__, fid)


def load_tokenizer(path="awni/lmx"):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    return transformers.AutoTokenizer.from_pretrained(path)


def save_checkpoint(save_dir, it, params, optimizer, config):
    checkpoint_dir = Path(save_dir)
    if it is not None:
        checkpoint_dir /= f"{it:012d}"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    save_config(checkpoint_dir, config)

    mx.save_safetensors(
        str(checkpoint_dir / "model.safetensors"),
        dict(tree_flatten(params)),
    )
    mx.save_safetensors(
        str(checkpoint_dir / "opt_state.safetensors"),
        dict(tree_flatten(optimizer.state)),
    )


@dataclasses.dataclass
class Metrics:
    step: int = 0
    tokens: int = 0
    train_loss: list = dataclasses.field(default_factory=list)
    grad_norm: list = dataclasses.field(default_factory=list)
    its_per_sec: float = 0.0
    toks_per_sec: float = 0.0
    valid_loss: float = None
    valid_ppl: float = None
    eval_core: float = None

    def to_list(self):
        metrics = [
            ("step", self.step),
            ("train_loss", np.mean(self.train_loss).item()),
            ("grad_norm", np.mean(self.grad_norm).item()),
            ("its_per_sec", self.its_per_sec),
            ("toks_per_sec", self.toks_per_sec),
            ("tokens", self.tokens),
        ]

        if self.valid_loss is not None:
            metrics.append(("valid_loss", self.valid_loss))
            metrics.append(("valid_ppl", self.valid_ppl))

        if self.eval_core is not None:
            metrics.append(("eval_core", self.eval_core))

        return metrics


def log_metrics(metrics):
    if isinstance(metrics, list):
        list_metrics = metrics
    else:
        list_metrics = metrics.to_list()

    def to_str(val):
        if isinstance(val, float):
            return f"{val:.4f}"
        else:
            return repr(val)

    print(", ".join(f"{n}: {to_str(v)}" for n, v in list_metrics))
    metrics = dict(list_metrics)
    step = metrics.pop("step")
    wandb.log(metrics, step=step)


def init_distributed():

    rank = int(os.environ.get("MLX_RANK", "0"))
    world_size = int(os.environ.get("MLX_WORLD_SIZE", "1"))

    if world_size > 1:
        if rank == 0:
            print(f"Master host: {os.environ.get('NCCL_HOST_IP')}")
        print(f"Rank {rank} of {world_size} initialized.")
        mx.distributed.init(backend="nccl")
    return rank, world_size


def load_optimizer(config):
    warmup = optim.linear_schedule(
        0,
        config.learning_rate,
        config.warmup_steps,
    )
    decay = optim.linear_schedule(
        config.learning_rate,
        0,
        config.decay_steps,
    )
    lr_schedule = optim.join_schedules(
        [warmup, decay], [config.num_steps - config.decay_steps]
    )
    if config.optim == "adam":
        optimizer = optim.Adam(learning_rate=lr_schedule)
    elif config.optim == "adamw":
        optimizer = optim.AdamW(
            learning_rate=lr_schedule,
            weight_decay=config.get("weight_decay", 0),
        )
    elif config.optim == "sgd":
        optimizer = optim.SGD(learning_rate=lr_schedule)
    return optimizer


def load_model(config):
    model_type = config["model_type"]
    arch = importlib.import_module(f"models.{model_type}")
    model_args = arch.ModelArgs(
        **{
            k: v
            for k, v in config.items()
            if k in inspect.signature(arch.ModelArgs).parameters
        }
    )
    return arch.Model(model_args)


if __name__ == "__main__":

    tokenizer = load_tokenizer()
    print(f"Vocab size {len(tokenizer.vocab)}")
