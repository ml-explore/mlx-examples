import argparse
import time
from functools import partial

import kwt
import mlx.core as mx
import mlx.data as dx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.data.datasets import load_speechcommands
from mlx.data.features import mfsc

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument(
    "--arch",
    type=str,
    default="kwt1",
    choices=[f"kwt{d}" for d in [1, 2, 3]],
    help="model architecture",
)
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--cpu", action="store_true", help="use cpu only")


def prepare_dataset(batch_size, split, root=None):
    def normalize(x):
        return (x - x.mean()) / x.std()

    data = load_speechcommands(split=split, root=root)

    data_iter = (
        data.squeeze("audio")
        .key_transform(
            "audio",
            mfsc(
                40,
                16000,
                frame_size_ms=30,
                frame_stride_ms=10,
                high_freq=7600,
                low_freq=20,
            ),
        )
        .key_transform("audio", normalize)
        .shuffle()
        .batch(batch_size)
        .to_stream()
        .prefetch(4, 4)
    )
    return data_iter


def eval_fn(model, x, y):
    return mx.mean(mx.argmax(model(x), axis=1) == y)


def train_epoch(model, train_iter, optimizer, epoch):
    def train_step(model, x, y):
        output = model(x)
        loss = mx.mean(nn.losses.cross_entropy(output, y))
        acc = mx.mean(mx.argmax(output, axis=1) == y)
        return loss, acc

    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y):
        (loss, acc), grads = nn.value_and_grad(model, train_step)(model, x, y)
        optimizer.update(model, grads)
        return loss, acc

    losses = []
    accs = []
    samples_per_sec = []

    model.train(True)
    for batch_counter, batch in enumerate(train_iter):
        x = mx.array(batch["audio"])
        y = mx.array(batch["label"])
        tic = time.perf_counter()
        loss, acc = step(x, y)
        mx.eval(state)
        toc = time.perf_counter()
        loss = loss.item()
        acc = acc.item()
        losses.append(loss)
        accs.append(acc)
        throughput = x.shape[0] / (toc - tic)
        samples_per_sec.append(throughput)
        if batch_counter % 25 == 0:
            print(
                " | ".join(
                    (
                        f"Epoch {epoch:02d} [{batch_counter:03d}]",
                        f"Train loss {loss:.3f}",
                        f"Train acc {acc:.3f}",
                        f"Throughput: {throughput:.2f} samples/second",
                    )
                )
            )

    mean_tr_loss = mx.mean(mx.array(losses))
    mean_tr_acc = mx.mean(mx.array(accs))
    samples_per_sec = mx.mean(mx.array(samples_per_sec))
    return mean_tr_loss, mean_tr_acc, samples_per_sec


def test_epoch(model, test_iter):
    model.train(False)
    accs = []
    throughput = []
    for batch_counter, batch in enumerate(test_iter):
        x = mx.array(batch["audio"])
        y = mx.array(batch["label"])
        tic = time.perf_counter()
        acc = eval_fn(model, x, y)
        accs.append(acc.item())
        toc = time.perf_counter()
        throughput.append(x.shape[0] / (toc - tic))
    mean_acc = mx.mean(mx.array(accs))
    mean_throughput = mx.mean(mx.array(throughput))
    return mean_acc, mean_throughput


def main(args):
    mx.random.seed(args.seed)

    model = getattr(kwt, args.arch)()

    print("Number of params: {:0.04f} M".format(model.num_params() / 1e6))

    optimizer = optim.SGD(learning_rate=args.lr, momentum=0.9, weight_decay=1e-4)

    train_data = prepare_dataset(args.batch_size, "train")
    val_data = prepare_dataset(args.batch_size, "validation")

    best_params = None
    best_acc = 0.0
    best_epoch = 0
    for epoch in range(args.epochs):
        tr_loss, tr_acc, throughput = train_epoch(model, train_data, optimizer, epoch)
        print(
            " | ".join(
                (
                    f"Epoch: {epoch}",
                    f"avg. Train loss {tr_loss.item():.3f}",
                    f"avg. Train acc {tr_acc.item():.3f}",
                    f"Throughput: {throughput.item():.2f} samples/sec",
                )
            )
        )

        val_acc, val_throughput = test_epoch(model, val_data)
        print(
            f"Epoch: {epoch} | Val acc {val_acc.item():.3f} | Throughput: {val_throughput.item():.2f} samples/sec"
        )

        if val_acc >= best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_params = model.parameters()
    print(f"Testing best model from epoch {best_epoch}")
    model.update(best_params)
    test_data = prepare_dataset(args.batch_size, "test")
    test_acc, _ = test_epoch(model, test_data)
    print(f"Test acc -> {test_acc.item():.3f}")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.cpu:
        mx.set_default_device(mx.cpu)
    main(args)
