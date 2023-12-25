# Copyright © 2023 Apple Inc.

import functools
import time

import jax
import jax.numpy as jnp

import mnist


def init_model(key, num_layers, input_dim, hidden_dim, output_dim):
    params = []
    layer_sizes = [hidden_dim] * num_layers
    for idim, odim in zip([input_dim] + layer_sizes, layer_sizes + [output_dim]):
        key, wk = jax.random.split(key, 2)
        W = 1e-2 * jax.random.normal(wk, (idim, odim))
        b = jnp.zeros((odim,))
        params.append((W, b))
    return params


def feed_forward(params, X):
    for W, b in params[:-1]:
        X = jnp.maximum(X @ W + b, 0)
    W, b = params[-1]
    return X @ W + b


def loss_fn(params, X, y):
    logits = feed_forward(params, X)
    logits = jax.nn.log_softmax(logits, 1)
    return -jnp.mean(logits[jnp.arange(y.size), y])


@jax.jit
def eval_fn(params, X, y):
    logits = feed_forward(params, X)
    return jnp.mean(jnp.argmax(logits, axis=1) == y)


def batch_iterate(key, batch_size, X, y):
    perm = jax.random.permutation(key, y.size)
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids], y[ids]


if __name__ == "__main__":
    seed = 0
    num_layers = 2
    hidden_dim = 32
    num_classes = 10
    batch_size = 256
    num_epochs = 10
    learning_rate = 1e-1
    dataset = "mnist"

    # Load the data
    train_images, train_labels, test_images, test_labels = getattr(mnist, dataset)()
    # Load the model
    key, subkey = jax.random.split(jax.random.PRNGKey(seed))
    params = init_model(
        subkey, num_layers, train_images.shape[-1], hidden_dim, num_classes
    )

    loss_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    update_fn = jax.jit(
        functools.partial(jax.tree_map, lambda p, g: p - learning_rate * g)
    )

    for e in range(num_epochs):
        tic = time.perf_counter()
        key, subkey = jax.random.split(key)
        for X, y in batch_iterate(subkey, batch_size, train_images, train_labels):
            loss, grads = loss_and_grad_fn(params, X, y)
            params = update_fn(params, grads)
        accuracy = eval_fn(params, test_images, test_labels)
        toc = time.perf_counter()
        print(
            f"Epoch {e}: Test accuracy {accuracy.item():.3f},"
            f" Time {toc - tic:.3f} (s)"
        )
