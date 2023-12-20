# Copyright © 2023 Apple Inc.

import math
import time

import datasets
import numpy as np
import tensorflow as tf


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


def iterate_batches(batch_size, context_size, dataset):
    inputs, targets = to_samples(context_size, dataset)
    s = 0
    while True:
        if s == 0:
            # Reset permutation:
            perm = np.random.permutation(inputs.shape[0])
        ids = perm[s : s + batch_size]
        yield inputs[ids], targets[ids]
        s += batch_size
        if s + batch_size >= inputs.shape[0]:
            s = 0


def create_additive_causal_mask(N):
    indices = tf.range(N)
    mask = tf.reshape(indices, (-1, 1)) < tf.reshape(indices, (1, -1))
    return tf.cast(mask, tf.dtypes.float32) * -1e9


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, model_dims, context_size):
        super().__init__()
        self.Wq = tf.keras.layers.Dense(model_dims, use_bias=False)
        self.Wk = tf.keras.layers.Dense(model_dims, use_bias=False)
        self.Wv = tf.keras.layers.Dense(model_dims, use_bias=False)
        self.Wo = tf.keras.layers.Dense(model_dims, use_bias=False)
        self.causal_mask = create_additive_causal_mask(context_size)
        self.num_heads = num_heads
        self.head_dim = model_dims // num_heads
        self.scale = tf.constant(1.0 / math.sqrt(self.head_dim))

    def call(self, x):
        queries = self.Wq(x)
        keys = self.Wk(x)
        values = self.Wv(x)

        B, L, D = x.shape
        queries = tf.transpose(
            tf.reshape(queries, (B, L, self.num_heads, -1)), perm=(0, 2, 1, 3)
        )
        keys = tf.transpose(
            tf.reshape(keys, (B, L, self.num_heads, -1)), perm=(0, 2, 1, 3)
        )
        values = tf.transpose(
            tf.reshape(values, (B, L, self.num_heads, -1)), perm=(0, 2, 1, 3)
        )

        scores = (self.scale * queries) @ tf.transpose(keys, (0, 1, 3, 2))
        scores = tf.nn.softmax(scores + self.causal_mask, axis=-1)
        values = tf.matmul(scores, values)
        values_hat = tf.reshape(tf.transpose(values, perm=(0, 2, 1, 3)), (B, L, -1))

        return self.Wo(values_hat)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, model_dims, context_size):
        super().__init__()
        self._ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

        self._self_attn = SelfAttention(num_heads, model_dims, context_size)

        self._ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

        self._mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(4 * model_dims, activation="relu"),
                tf.keras.layers.Dense(model_dims),
            ]
        )

    def call(self, x):
        x = x + self._self_attn(self._ln1(x))
        x = x + self._mlp(self._ln2(x))
        return x


class TransformerLM(tf.keras.Model):
    def __init__(self, vocab_size, num_layers, num_heads, model_dims, context_size):
        super().__init__()

        self.embedding = tf.keras.layers.Embedding(vocab_size, model_dims)
        self.transformer = tf.keras.Sequential(
            [
                EncoderLayer(num_heads, model_dims, context_size)
                for _ in range(num_layers)
            ]
        )
        self.projection = tf.keras.layers.Dense(vocab_size)

    def call(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.projection(x)
        return x


def main(args, device):
    with tf.device(device):
        batch_size = args.batch_size
        context_size = args.context_size
        steps_per_eval = args.steps_per_eval
        steps_per_report = args.steps_per_report

        # Load vocab and dataset:
        vocab, train, valid, test = datasets.ptb()

        # Initialize model:
        transformer = TransformerLM(
            len(vocab), args.num_blocks, args.num_heads, args.dim, context_size
        )
        transformer.compile(
            optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=args.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )
        transformer.build((batch_size, context_size))
        nparams = sum(
            np.prod(p.shape) for p in transformer.trainable_weights[1:]
        )  # [1:] to skip the embedding
        print(f"Training a transformer with {nparams / 1024**2:.3f} M parameters")

        def eval_fn(dataset):
            inputs, targets = to_samples(context_size, dataset)
            loss = 0
            n_batches = 0
            for s in range(0, targets.shape[0], batch_size):
                if s + batch_size >= targets.shape[0]:
                    s = targets.shape[0] - 1 - batch_size
                bx, by = inputs[s : s + batch_size], targets[s : s + batch_size]
                bx, by = map(
                    lambda x: tf.convert_to_tensor(x, dtype=tf.dtypes.int32),
                    [bx, by],
                )
                loss += transformer.test_on_batch(bx, by)
                n_batches += 1
            return loss / n_batches

        train_iterator = iterate_batches(batch_size, context_size, train)
        losses = []
        tic = time.perf_counter()
        for it, (inputs, targets) in zip(range(args.num_iters), train_iterator):
            inputs, targets = map(
                lambda x: tf.convert_to_tensor(x, dtype=tf.dtypes.int32),
                [inputs, targets],
            )
            loss = transformer.train_on_batch(inputs, targets)
            losses.append(loss)
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
                val_loss = eval_fn(valid)
                toc = time.perf_counter()
                print(
                    f"Iter {it + 1}: "
                    f"Val loss {val_loss:.3f}, "
                    f"Val ppl {math.exp(val_loss):.3f}, "
                    f"Val took {(toc - tic):.3f}s, "
                )
                tic = time.perf_counter()

        if args.eval_test:
            test_loss = eval_fn(test)
            test_ppl = math.exp(test_loss)
            print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Train a decoder-only Transformer LM with MLX.")
    parser.add_argument("--gpu", action="store_true", help="Use the Metal back-end.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the RNGs.")
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
    main(args, device="/GPU:0" if args.gpu else "/CPU:0")
