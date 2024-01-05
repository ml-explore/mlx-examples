"""
Based on lora and performs supervised instruction finetuning
However, it but breaks out parameters into a YAML file and allows arbitrary training data (and prompt) formats

The configuration .yaml file is expected to be in the following format:

parameters:
    model: "..."
    num_tokens: 100
    [..]

Where each entry under parameters is the argparse version of the argumens originally provided to lora.py plus new ones

A module for a particular prompt syntax or training dataset format just needs to overide TrainingRecordHandler,
provide an instance of it to main, and run the script with a single argument which is a path to a YAML file with the
configuration parameters that were originally command-line arguments in lora.py

See mistral_supervised.py for an example

An epoch parameter was added, which determines the number of iterations if provided (the number needed for a
full pass of the data, i.e., an epoch).  If the value is -1, it is ignored and the iters parameter is used same as
before, except if iters is -1 then one epoch is performed

"""

import argparse
import json
import math
import time
from abc import ABC
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, List

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yaml
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from models import LoRALinear, Model, ModelArgs
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm


class TrainingRecordHandler(ABC):
    """
    Provides two methods for extracting the inputs and outputs (labels) from a supervised training dataset dictionary

    """

    def get_input(self, record) -> str:
        pass

    def get_output(self, record) -> str:
        pass


class Dataset:
    """
    Light-weight wrapper to hold data a jsonl file for use in training, validation, and testing
    """

    def __init__(self, path: Path):
        if not path.exists():
            self._data = None
        else:
            with open(path, "r") as f:
                self._data = json.load(f)

    def __getitem__(self, idx: int):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


CONFIG_DEFAULTS = {
    "num_tokens": 100,
    "write_every": 1,
    "prompt": None,
    "train": False,
    "data": "data/",
    "lora_layers": 16,
    "batch_size": 4,
    "iters": -1,
    "epochs": -1,
    "val_batches": 25,
    "learning_rate": 1e-5,
    "steps_per_report": 10,
    "steps_per_eval": 200,
    "resume_adapter_file": None,
    "adapter_file": "adapters.npz",
    "test": False,
    "test_batches": 500,
    "seed": 0,
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="LoRA supervised finetuning with Mistral"
    )
    parser.add_argument("filename", help="The YAML confguration file")
    with open(parser.parse_args().filename, "r") as file:
        config = yaml.safe_load(file)
        param_dict = config["parameters"]
        if "model" not in param_dict:
            raise SyntaxError('Missing required "model" parameter')
        for key, default in CONFIG_DEFAULTS.items():
            if key not in param_dict:
                param_dict[key] = default
    print("Configuration:", param_dict)
    return SimpleNamespace(**param_dict)


class Tokenizer:
    def __init__(self, model_path: str):
        assert Path(model_path).exists(), model_path
        self._model = SentencePieceProcessor(model_file=model_path)
        self._sep = "▁"
        assert self._model.vocab_size() == self._model.get_piece_size()

    def encode(self, s: str, bos: bool = True, eos: bool = False) -> List[int]:
        toks = (
            [self._model.bos_id(), *self._model.encode(s)]
            if bos
            else [*self._model.encode(s)]
        )
        if eos:
            toks.append(self.eos_id)
        return toks

    @property
    def eos_id(self) -> int:
        return self._model.eos_id()

    def decode(self, t: List[int]) -> str:
        out = self._model.decode(t)
        if t and self._model.id_to_piece(t[0])[0] == self._sep:
            return " " + out
        return out

    @property
    def vocab_size(self) -> int:
        return self._model.vocab_size()


def load(args):
    names = ("train", "valid", "test")
    train, valid, test = (Dataset(Path(args.data) / f"{n}.jsonl") for n in names)
    if args.train and len(train) < 0:
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


def loss(model: Model, inputs: mx.array, targets: mx.array, output_lengths):
    # Run model on inputs
    logits, _ = model(inputs)
    logits = logits.astype(mx.float32)

    # Mask padding tokens for output (prevent penalization of generating padding tokens)
    mask = mx.arange(targets.shape[1])[None, :] < output_lengths[:, None]

    # Calculate the loss and use mask to prevent penalizing the model for not recreating the padding suffix
    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    ce = ce.sum() / ntoks
    return ce, ntoks


def iterate_batches(
    dset: Dataset,
    tokenizer: Tokenizer,
    batch_size: int,
    handler: TrainingRecordHandler,
    train: bool = True,
) -> Iterator[tuple[mx.array, mx.array, mx.array]]:
    """
    Continuously generate a tuple of 2 batch_size x N matrices (each an mx.array) and a vector (also an mx.array)
    of size batch_size.

    N is the length of the longest tokenization of the input or output of a record from dset.

    These matrices are the tokenizations of the input and output of data records respectively.


    Each row of the matrices has a zero-padding suffix beyond the length of the token sequence up to N (or none
    for the longest token sequence).

    The records are pulled from dset in random order (if train is True or in original order otherwise)
    in groups of batch_size.

    The vector is the length of each output token sequence in the batch (excluding the zero-padding suffix)
    """
    while True:
        indices = np.arange(len(dset))
        if train:
            # Shuffle order of batches pulled from dataset
            indices = np.random.permutation(indices)
        # Collect batches of size batch_size from dataset either in original order or (if train is False)
        # shuffled
        for i in range(0, len(indices) - batch_size + 1, batch_size):
            # Extract Mistral prompt and output (labels) from the training data batch
            input_batch = [
                handler.get_input(dset[indices[i + j]]) for j in range(batch_size)
            ]
            output_batch = [
                handler.get_output(dset[indices[i + j]]) for j in range(batch_size)
            ]

            # Tokenize the input and output separately, with BOS only for input and EOS only for output
            input_batch = [tokenizer.encode(record) for record in input_batch]
            output_batch = [
                tokenizer.encode(record, bos=False, eos=True) for record in output_batch
            ]

            # Collect the token lengths for use in adding zero padding for input and output.  The latter is used
            # For the mask used when calculating the loss
            input_lengths = [len(x) for x in input_batch]
            output_lengths = [len(x) for x in output_batch]

            # Calculate maximum token sequence width from both input and output to use as the same width for
            # input and output batch array
            max_width = max(max(input_lengths), max(output_lengths))

            batch_input_arr = np.zeros((batch_size, max_width), np.int32)
            for j in range(batch_size):
                batch_input_arr[j, : input_lengths[j]] = input_batch[j]
            input_batch = mx.array(batch_input_arr)
            # input_batch is now an MLX array where each row corresponds to a record from the batch
            # and each item comprises the tokenization of the input zero padded up to the length
            # of the longest input token sequence

            # The same is done for the output
            batch_output_arr = np.zeros((batch_size, max_width), np.int32)
            for j in range(batch_size):
                batch_output_arr[j, : output_lengths[j]] = output_batch[j]
            output_batch = mx.array(batch_output_arr)

            yield input_batch, output_batch, mx.array(output_lengths)

        if not train:
            break


def evaluate(model, dataset, loss, tokenizer, batch_size, num_batches, handler):
    all_losses = []
    ntokens = 0
    for it, batch in zip(
        range(num_batches),
        iterate_batches(dataset, tokenizer, batch_size, handler),
    ):
        losses, toks = loss(model, *batch)
        all_losses.append((losses * toks).item())
        ntokens += toks.item()

    return np.sum(all_losses) / ntokens


def train(model, train_set, val_set, optimizer, loss, tokenizer, args, handler):
    # Create value and grad function for loss
    loss_value_and_grad = nn.value_and_grad(model, loss)

    losses = []
    n_tokens = 0

    # Main training loop
    start = time.perf_counter()

    # The number of steps for 1 epoch
    epoch_num_steps = (len(train_set) + args.batch_size - 1) // args.batch_size

    if args.epochs == -1:
        num_iterations = epoch_num_steps if args.iters == -1 else args.iters
    else:
        num_iterations = epoch_num_steps * args.epochs

    pbar = tqdm(total=epoch_num_steps)
    print(
        f"{num_iterations:,} iterations at {epoch_num_steps:,} iterations per epoch on a dataset of "
        f"{len(train_set):,} records, {args.batch_size} at a time."
    )
    for it, batch in zip(
        range(num_iterations),
        iterate_batches(train_set, tokenizer, args.batch_size, handler, train=True),
    ):
        # Forward and backward pass
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)

        # Model update
        optimizer.update(model, grad)
        mx.eval(model.parameters(), optimizer.state, lvalue)

        # Record loss
        losses.append(lvalue.item())
        n_tokens += toks.item()

        # Report training loss if needed
        if (it + 1) % args.steps_per_report == 0:
            train_loss = np.mean(losses)

            stop = time.perf_counter()
            print(
                f"Iter {it + 1}: Train loss {train_loss:.3f}, "
                f"It/sec {args.steps_per_report / (stop - start):.3f}, "
                f"Tokens/sec {float(n_tokens) / (stop - start):.3f}"
            )
            losses = []
            n_tokens = 0
            start = time.perf_counter()

        # Report validation loss if needed
        if it == 0 or (it + 1) % args.steps_per_eval == 0:
            stop = time.perf_counter()
            val_loss = evaluate(
                model,
                val_set,
                loss,
                tokenizer,
                args.batch_size,
                args.val_batches,
                handler,
            )
            print(
                f"Iter {it + 1}: "
                f"Val loss {val_loss:.3f}, "
                f"Val took {(time.perf_counter() - stop):.3f}s"
            )

            start = time.perf_counter()
        pbar.update(1)


def generate(model, prompt, tokenizer, args):
    print(args.prompt, end="", flush=True)
    prompt = mx.array(tokenizer.encode(args.prompt))

    def generate_step():
        temp = args.temp

        def sample(logits):
            if temp == 0:
                return mx.argmax(logits, axis=-1)
            else:
                return mx.random.categorical(logits * (1 / temp))

        logits, cache = model(prompt[None])
        y = sample(logits[:, -1, :])
        yield y

        while True:
            logits, cache = model(y[:, None], cache)
            y = sample(logits.squeeze(1))
            yield y

    tokens = []
    for token, _ in zip(generate_step(), range(args.num_tokens)):
        tokens.append(token)

        if (len(tokens) % 10) == 0:
            mx.eval(tokens)
            s = tokenizer.decode([t.item() for t in tokens])
            print(s, end="", flush=True)
            tokens = []

    mx.eval(tokens)
    s = tokenizer.decode([t.item() for t in tokens])
    print(s, flush=True)


def load_model(folder: str, dtype=mx.float16):
    model_path = Path(folder)
    tokenizer = Tokenizer(str(model_path / "tokenizer.model"))
    with open(model_path / "params.json", "r") as f:
        config = json.loads(f.read())
        if config.get("vocab_size", -1) < 0:
            config["vocab_size"] = tokenizer.vocab_size
        model_args = ModelArgs(**config)
    weights = mx.load(str(model_path / "weights.npz"))
    weights = tree_unflatten(list(weights.items()))
    weights = tree_map(lambda p: p.astype(dtype), weights)
    model = Model(model_args)
    model.update(weights)
    return model, tokenizer


def main(handler: TrainingRecordHandler):
    global args, model, l
    args = build_parser()
    np.random.seed(args.seed)
    print("Loading pretrained model")
    model, tokenizer = load_model(args.model)
    # Freeze all layers other than LORA linears
    model.freeze()
    for l in model.layers[-args.lora_layers :]:
        l.attention.wq = LoRALinear.from_linear(l.attention.wq)
        l.attention.wv = LoRALinear.from_linear(l.attention.wv)
    p = sum(v.size for _, v in tree_flatten(model.parameters())) / 10**6
    print(f"Total parameters {p:.3f}M")
    p = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
    print(f"Trainable parameters {p:.3f}M")
    print("Loading datasets")
    train_set, valid_set, test_set = load(args)
    # Resume training the given adapters.
    if args.resume_adapter_file is not None:
        print(f"Loading pretrained adapters from {args.resume_adapter_file}")
        model.load_weights(args.resume_adapter_file)
    if args.train:
        print("Training")
        opt = optim.Adam(learning_rate=args.learning_rate)

        # Train model
        train(model, train_set, valid_set, opt, loss, tokenizer, args, handler)

        # Save adapter weights
        mx.savez(args.adapter_file, **dict(tree_flatten(model.trainable_parameters())))
    # Load the LoRA adapter weights which we assume should exist by this point
    model.load_weights(args.adapter_file)
    if args.test:
        print("Testing")

        test_loss = evaluate(
            model,
            test_set,
            loss,
            tokenizer,
            args.batch_size,
            args.test_batches,
            handler,
        )
        test_ppl = math.exp(test_loss)

        print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")
    if args.prompt is not None:
        print("Generating")
        generate(model, args.prompt, tokenizer, args)
