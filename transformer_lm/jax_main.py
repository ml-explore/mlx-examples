# Copyright © 2023 Apple Inc.

import functools
import math
import time
from collections import namedtuple

import datasets
import jax
import jax.numpy as jnp
import numpy as np
from tree_utils import tree_flatten

"""
Some TODOs for this model:
    - Positional encodings
    - Dropout
    - Adam optimizer
    - Option for bigger datasets (wikitext / librispeech text < c4 < ...)
"""

RuntimeConfig = namedtuple("RuntimeConfig", "num_heads")


def embedding_init(key, num_embeddings, embed_dim):
    return jax.random.uniform(
        key, (num_embeddings, embed_dim), minval=-1e-1, maxval=1e-1
    )


def embedding_apply(params, X):
    return params.take(X, axis=0)


def dense_init(key, in_dim, out_dim, bias=True):
    k1, k2 = jax.random.split(key)
    scale = math.sqrt(1 / in_dim)
    params = [jax.random.uniform(k1, (in_dim, out_dim), minval=-scale, maxval=scale)]
    if bias:
        params.append(jax.random.uniform(k2, (out_dim,), minval=-scale, maxval=scale))
    return params


def dense_apply(params, X):
    X = X @ params[0]
    if len(params) == 2:
        X = X + params[1]
    return X


def layernorm_init(key, dim):
    return [jnp.zeros((dim,)), jnp.ones((dim,))]


def layernorm_apply(params, X, epsilon=1e-6):
    means = jnp.mean(X, axis=-1, keepdims=True)
    var = jnp.var(X, axis=-1, keepdims=True)
    X = (X - means) / jnp.sqrt(var + epsilon)
    beta, gamma = params
    return gamma * X + beta


def mlpblock_init(key, dim):
    k1, k2 = jax.random.split(key)
    return {
        "dense1": dense_init(k1, dim, 4 * dim),
        "dense2": dense_init(k2, 4 * dim, dim),
    }


def mlpblock_apply(params, X):
    X = dense_apply(params["dense1"], X)
    X = jnp.maximum(X, 0)
    # TODO dropout option here
    return dense_apply(params["dense2"], X)


def selfattention_init(key, dim):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return {
        "Q": dense_init(k1, dim, dim, bias=False),
        "K": dense_init(k2, dim, dim, bias=False),
        "V": dense_init(k3, dim, dim, bias=False),
        "out": dense_init(k4, dim, dim, bias=False),
    }


def selfattention_apply(params, num_heads, X, mask):
    queries = dense_apply(params["Q"], X)
    keys = dense_apply(params["K"], X)
    values = dense_apply(params["V"], X)

    B, L, D = queries.shape
    queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
    keys = keys.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)

    # Dimensions are [batch x num heads x sequence x hidden dim]
    scale = math.sqrt(1 / queries.shape[-1])
    scores = (queries * scale) @ jnp.transpose(keys, (0, 1, 3, 2))
    scores = jax.nn.softmax(scores + mask, axis=-1)
    values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

    return dense_apply(params["out"], values_hat)


def transformer_init(key, token_set_size, num_blocks, dim):
    key, ek = jax.random.split(key)
    params = {"embedding": embedding_init(ek, token_set_size, dim)}
    transformer_blocks = []
    for b in range(num_blocks):
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        transformer_blocks.append(
            {
                "ln1": layernorm_init(k1, dim),
                "ln2": layernorm_init(k2, dim),
                "selfattention": selfattention_init(k3, dim),
                "mlpblock": mlpblock_init(k4, dim),
            }
        )
    params["transformer_blocks"] = transformer_blocks
    params["output"] = dense_init(key, dim, token_set_size)
    return params


def create_additive_causal_mask(N):
    indices = jnp.arange(N)
    mask = jnp.reshape(indices, (-1, 1)) < jnp.reshape(indices, (1, -1))
    # usually inf but 1e9 is as good and softmax(full(1e9)) != nan
    mask = mask.astype(jnp.float32) * -1e9
    return mask


def transformer_apply(params, static_params, inputs):
    mask = create_additive_causal_mask(inputs.shape[1])
    X = embedding_apply(params["embedding"], inputs)
    for block in params["transformer_blocks"]:
        out = layernorm_apply(block["ln1"], X)
        out = selfattention_apply(
            block["selfattention"], static_params.num_heads, out, mask
        )
        X = X + out
        out = layernorm_apply(block["ln2"], X)
        out = mlpblock_apply(block["mlpblock"], out)
        X = X + out
    return dense_apply(params["output"], X)


@functools.partial(jax.jit, static_argnames=["static_params", "reduce"])
def loss_fn(params, static_params, inputs, targets, reduce=True):
    logits = transformer_apply(params, static_params, inputs)
    logits = jax.nn.log_softmax(logits, axis=-1)
    sample_indices = jnp.arange(targets.shape[0])[:, None]
    token_indices = jnp.arange(targets.shape[1])[None, :]
    losses = -logits[sample_indices, token_indices, targets]
    return jnp.mean(losses) if reduce else losses.mean(-1)


def to_samples(context_size, dataset):
    tokens = dataset.size
    window_size = context_size + 1  # include target
    samples = tokens - window_size + 1
    X = np.lib.stride_tricks.as_strided(
        dataset,
        shape=(samples, window_size),
        strides=(dataset.itemsize, dataset.itemsize),
    )
    return X[:, :-1], X[:, 1:]


def iterate_batches(key, batch_size, context_size, dataset):
    inputs, targets = to_samples(context_size, dataset)
    s = 0
    while True:
        if s == 0:
            # Reset permutation:
            key, subkey = jax.random.split(key)
            perm = jax.random.permutation(subkey, inputs.shape[0])
        ids = perm[s : s + batch_size]
        yield inputs[ids], targets[ids]
        s += batch_size
        if s >= inputs.shape[0]:
            s = 0


def main(args):
    batch_size = args.batch_size
    context_size = args.context_size
    steps_per_eval = args.steps_per_eval
    steps_per_report = args.steps_per_report
    config = RuntimeConfig(args.num_heads)

    # Load vocab and dataset:
    vocab, train, valid, test = datasets.ptb()

    # Initialize model:
    key, subkey = jax.random.split(jax.random.PRNGKey(args.seed))
    params = transformer_init(subkey, len(vocab), args.num_blocks, args.dim)
    nparams = sum(x.size for k, x in tree_flatten(params) if "embedding" not in k)
    print(f"Training a transformer with {nparams / 1024**2:.3f} M parameters")

    loss_and_grad_fn = jax.jit(
        jax.value_and_grad(loss_fn), static_argnames=["static_params"]
    )
    update_fn = jax.jit(
        functools.partial(jax.tree_map, lambda p, g: p - args.learning_rate * g)
    )

    def eval_fn(params, dataset):
        inputs, targets = to_samples(context_size, dataset)
        loss = 0
        for s in range(0, targets.shape[0], batch_size):
            bx, by = inputs[s : s + batch_size], targets[s : s + batch_size]
            losses = loss_fn(params, config, bx, by, reduce=False)
            loss += jnp.sum(losses)
        return loss / len(targets)

    train_iterator = iterate_batches(subkey, batch_size, context_size, train)
    losses = []
    tic = time.perf_counter()
    for it, (inputs, targets) in zip(range(args.num_iters), train_iterator):
        loss, grads = loss_and_grad_fn(params, config, inputs, targets)
        losses.append(loss.item())
        params = update_fn(params, grads)
        if (it + 1) % steps_per_report == 0:
            train_loss = np.mean(losses)
            toc = time.perf_counter()
            print(
                f"Iter {it + 1}: Train loss {train_loss:.3f}, "
                f"It/sec {steps_per_report / (toc - tic):.3f}"
            )
            losses = []
            tic = time.perf_counter()

        if (it + 1) % steps_per_eval == 0:
            val_loss = eval_fn(params, valid)
            toc = time.perf_counter()
            print(
                f"Iter {it + 1}: "
                f"Val loss {val_loss:.3f}, "
                f"Val ppl {math.exp(val_loss):.3f}, "
                f"Val took {(toc - tic):.3f}s, "
            )
            tic = time.perf_counter()

    if args.eval_test:
        test_loss = eval_fn(params, test)
        test_ppl = math.exp(test_loss)
        print(f"Test loss {test_loss.item():.3f}, Test ppl {test_ppl:.3f}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Train a decoder-only Transformer LM with Jax.")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for numpy and Jax RNGs."
    )
    parser.add_argument(
        "--context_size",
        type=int,
        default=1024,
        help="Context size in tokens of the model.",
    )
    parser.add_argument(
        "--num_blocks", type=int, default=12, help="Number of Transformer blocks."
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=1024,
        help="Dimensionality of embeddings and hidden layers.",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=16,
        help="Number of heads used for multi-head attention",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Minibatch size.")
    parser.add_argument(
        "--num_iters", type=int, default=100000, help="Iterations to train for."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="SGD learning rate."
    )
    parser.add_argument(
        "--steps_per_report",
        type=int,
        default=10,
        help="Number of training steps between loss reporting.",
    )
    parser.add_argument(
        "--steps_per_eval",
        type=int,
        default=1000,
        help="Number of training steps between validations.",
    )
    parser.add_argument(
        "--eval_test",
        action="store_true",
        help="Evaluate on the test set after training",
    )
    args = parser.parse_args()
    main(args)
