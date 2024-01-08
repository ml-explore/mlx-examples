# Copyright © 2023 Apple Inc.

import argparse
import json
import math
import time
from pathlib import Path
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from models import LoRALinear, Model, ModelArgs
from sentencepiece import SentencePieceProcessor

class Tokenizer:
    def __init__(self, model_path: str):
        assert Path(model_path).exists(), model_path
        self._model = SentencePieceProcessor(model_file=model_path)
        self._sep = "▁"
        assert self._model.vocab_size() == self._model.get_piece_size()

    def encode(self, s: str, eos: bool = False) -> List[int]:
        toks = [self._model.bos_id(), *self._model.encode(s)]
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
        return len(self._data)


def load(data_path, train_or_test: str = "train"):
    names = ("train", "valid", "test")
    train, valid, test = (Dataset(Path(data_path) / f"{n}.jsonl") for n in names)
    if train_or_test == "train" and len(train) == 0:
        raise ValueError(
            "Training set not found or empty. Must provide training set for fine-tuning."
        )
    if train_or_test == "train" and len(valid) == 0:
        raise ValueError(
            "Validation set not found or empty. Must provide validation set for fine-tuning."
        )
    if train_or_test == "test" and len(test) == 0:
        raise ValueError(
            "Test set not found or empty. Must provide test set for evaluation."
        )
    return train, valid, test


def loss(model, inputs, targets, lengths):
    # Run model on inputs
    logits, _ = model(inputs)
    logits = logits.astype(mx.float32)

    # Mask padding tokens
    length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

    # Calculate the loss
    ce = nn.losses.cross_entropy(logits, targets) * length_mask
    ntoks = length_mask.sum()
    ce = ce.sum() / ntoks
    return ce, ntoks


def iterate_batches(dset, tokenizer, batch_size, train=False):
    # Shuffle indices
    while True:
        indices = np.arange(len(dset))
        if train:
            indices = np.random.permutation(indices)

        # Collect batches from dataset
        for i in range(0, len(indices) - batch_size + 1, batch_size):
            # Encode batch
            batch = [
                tokenizer.encode(dset[indices[i + j]], eos=True)
                for j in range(batch_size)
            ]
            lengths = [len(x) for x in batch]
            
            # Check if any sequence is longer than 2048 tokens
            if max(lengths) > 2048:
                print("Warning: Some sequences are longer than 2048 tokens. Consider pre-splitting your data to save memory.")

            # Pad to the max length
            batch_arr = np.zeros((batch_size, max(lengths)), np.int32)
            for j in range(batch_size):
                batch_arr[j, : lengths[j]] = batch[j]
            batch = mx.array(batch_arr)
            yield batch[:, :-1], batch[:, 1:], mx.array(lengths)

        if not train:
            break


def evaluate(model, dataset, loss, tokenizer, batch_size, num_batches):
    all_losses = []
    ntokens = 0
    for it, batch in zip(
        range(num_batches),
        iterate_batches(dataset, tokenizer, batch_size),
    ):
        losses, toks = loss(model, *batch)
        all_losses.append((losses * toks).item())
        ntokens += toks.item()

    return np.sum(all_losses) / ntokens


def train(model, train_set, val_set, optimizer, loss, tokenizer, iters, batch_size, val_batches, steps_per_report, steps_per_eval):
    # Create value and grad function for loss
    loss_value_and_grad = nn.value_and_grad(model, loss)

    losses = []
    n_tokens = 0

    # Main training loop
    start = time.perf_counter()
    for it, batch in zip(
        range(iters),
        iterate_batches(train_set, tokenizer, batch_size, train=True),
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
        if (it + 1) % steps_per_report == 0:
            train_loss = np.mean(losses)

            stop = time.perf_counter()
            print(
                f"Iter {it + 1}: Train loss {train_loss:.3f}, "
                f"It/sec {steps_per_report / (stop - start):.3f}, "
                f"Tokens/sec {float(n_tokens) / (stop - start):.3f}"
            )
            losses = []
            n_tokens = 0
            start = time.perf_counter()

        # Report validation loss if needed
        if it == 0 or (it + 1) % steps_per_eval == 0:
            stop = time.perf_counter()
            val_loss = evaluate(
                model, val_set, loss, tokenizer, batch_size, val_batches
            )
            print(
                f"Iter {it + 1}: "
                f"Val loss {val_loss:.3f}, "
                f"Val took {(time.perf_counter() - stop):.3f}s"
            )

            start = time.perf_counter()


def generate(model, prompt, tokenizer, temp: float = 0.8, num_tokens: int = 100):
    print(prompt, end="", flush=True)
    prompt = mx.array(tokenizer.encode(prompt))

    def generate_step():

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
    for token, _ in zip(generate_step(), range(num_tokens)):
        tokens.append(token)

        if (len(tokens) % 10) == 0:
            mx.eval(tokens)
            s = tokenizer.decode([t.item() for t in tokens])
            print(s, end="", flush=True)
            tokens = []

    mx.eval(tokens)
    s = tokenizer.decode([t.item() for t in tokens])
    # print(s, flush=True)
    # returning just in case we need that
    # TODO: why does s return an empty string?
    return s

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

def prepare_for_training(model_path, data_path: str = "data/", seed: int = 0, lora_layers: int = 16, train_or_test: str = "train"):
    np.random.seed(seed)

    print("Loading pretrained model")
    model, tokenizer = load_model(model_path)

    print("Loading datasets")
    train_set, valid_set, test_set = load(data_path, train_or_test)

    if train_or_test == "train":
        # Freeze all layers other than LORA linears
        model.freeze()
        for l in model.layers[-lora_layers :]:
            l.attention.wq = LoRALinear.from_linear(l.attention.wq)
            l.attention.wv = LoRALinear.from_linear(l.attention.wv)

        p = sum(v.size for _, v in tree_flatten(model.parameters())) / 10**6
        print(f"Total parameters {p:.3f}M")
        p = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
        print(f"Trainable parameters {p:.3f}M")
        return model, tokenizer, train_set, valid_set
    elif train_or_test == "test":
        return model, tokenizer, test_set
    # elif train_or_test == "generate":
    #     return model, tokenizer
    else:
        raise ValueError(f"Unknown train_or_test {train_or_test}")


def run_lora_finetuning(model_path: str, data_path: str = "data/", lora_layers: int = 16, batch_size: int = 4, iters: int = 1000, seed: int = 0, 
                        resume_adapter_file: str = None, adapter_file: str = "adapters.npz", learning_rate: float = 1e-5,
                        val_batches: int = 25, steps_per_report: int = 10, steps_per_eval: int = 200):
    """
    Fine-tune the LoRA model.

    Parameters:
        model (str): A path to the model files containing the tokenizer, weights, config.
        data_path (str): Directory with {train, valid, test}.jsonl files.
        lora_layers (int): Number of layers to fine-tune. Default is 16.
        batch_size (int): Minibatch size. Default is 4.
        iters (int): Iterations to train for. Default is 1000.
    """
    # Training logic goes here
    model, tokenizer, train_set, val_set = prepare_for_training(model_path, data_path, seed, lora_layers, train_or_test="train")
    # Resume training the given adapters.
    if resume_adapter_file is not None:
        print(f"Loading pretrained adapters from {resume_adapter_file}")
        model.load_weights(resume_adapter_file)

    print("Training")
    # TODO: make optimizer a param maybe?
    opt = optim.Adam(learning_rate=learning_rate)

    # Train model
    train(model, train_set, val_set, opt, loss, tokenizer, iters, batch_size, val_batches, steps_per_report, steps_per_eval)

    # Save adapter weights
    mx.savez(adapter_file, **dict(tree_flatten(model.trainable_parameters())))

def run_lora_test(model_path, data_path: str = "data/", adapter_file: str = "adapters.npz", test_batches: int = 500, batch_size: int = 4):
    
    print("Testing")
    model, tokenizer, test_set = prepare_for_training(model_path, data_path, train_or_test="test")
    model.load_weights(adapter_file)
    
    test_loss = evaluate(
        model,
        test_set,
        loss,
        tokenizer,
        batch_size,
        num_batches=test_batches,
    )
    test_ppl = math.exp(test_loss)

    print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")
    return {"test_loss": test_loss, "test_ppl": test_ppl}


def run_lora_generate(model_path: str, num_tokens: int = 100, temp: float = 0.8, adapter_file: str = "adapters.npz", prompt: str = None):
    """
    Generate text using the LoRA model.

    Parameters:
        model (str): A path to the model files containing the tokenizer, weights, config.
        num_tokens (int): How many tokens to generate. Default is 100.
        write_every (int): After how many tokens to detokenize. Default is 1.
        temp (float): The sampling temperature. Default is 0.8.
        prompt (str): The prompt for generation. Default is None.
        val_batches (int): Number of validation batches, -1 uses the entire validation set. Default is 25.
        learning_rate (float): Adam learning rate. Default is 1e-5.
        steps_per_report (int): Number of training steps between loss reporting. Default is 10.
        steps_per_eval (int): Number of training steps between validations. Default is 200.
        resume_adapter_file (str): Load path to resume training with the given adapter weights. Default is None.
        adapter_file (str): Save/load path for the trained adapter weights. Default is "adapters.npz".
        test (bool): Evaluate on the test set after training. Default is False.
        test_batches (int): Number of test set batches, -1 uses the entire test set. Default is 500.
        seed (int): The PRNG seed. Default is 0.
    """
    # Generation logic goes here
    print("Generating")
    model, tokenizer = load_model(model_path)
    if adapter_file is not None:
        model.load_weights(adapter_file)
    else:
        raise ValueError("Must provide adapter_file to generate text.")
    return generate(model, prompt, tokenizer, temp, num_tokens)