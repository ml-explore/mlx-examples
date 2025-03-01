import argparse
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import resnet
from dataset import get_cifar10

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument(
    "--arch",
    type=str,
    default="resnet20",
    choices=[f"resnet{d}" for d in [20, 32, 44, 56, 110, 1202]],
    help="model architecture",
)
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--cpu", action="store_true", help="use cpu only")


def print_zero(group, *args, **kwargs):
    if group.rank() != 0:
        return
    flush = kwargs.pop("flush", True)
    print(*args, **kwargs, flush=flush)


def eval_fn(model, inp, tgt):
    return mx.mean(mx.argmax(model(inp), axis=1) == tgt)


def train_epoch(model, train_iter, optimizer, epoch):
    def train_step(model, inp, tgt):
        output = model(inp)
        loss = mx.mean(nn.losses.cross_entropy(output, tgt))
        acc = mx.mean(mx.argmax(output, axis=1) == tgt)
        return loss, acc

    world = mx.distributed.init()
    losses = 0
    accuracies = 0
    samples_per_sec = 0
    count = 0

    def average_stats(stats, count):
        if world.size() == 1:
            return [s / count for s in stats]

        with mx.stream(mx.cpu):
            stats = mx.distributed.all_sum(mx.array(stats))
            count = mx.distributed.all_sum(count)
            return (stats / count).tolist()

    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(inp, tgt):
        train_step_fn = nn.value_and_grad(model, train_step)
        (loss, acc), grads = train_step_fn(model, inp, tgt)
        grads = nn.utils.average_gradients(grads)
        optimizer.update(model, grads)
        return loss, acc

    for batch_counter, batch in enumerate(train_iter):
        x = mx.array(batch["image"])
        y = mx.array(batch["label"])
        tic = time.perf_counter()
        loss, acc = step(x, y)
        mx.eval(loss, acc, state)
        toc = time.perf_counter()
        losses += loss.item()
        accuracies += acc.item()
        samples_per_sec += x.shape[0] / (toc - tic)
        count += 1
        if batch_counter % 10 == 0:
            l, a, s = average_stats(
                [losses, accuracies, world.size() * samples_per_sec],
                count,
            )
            print_zero(
                world,
                " | ".join(
                    (
                        f"Epoch {epoch:02d} [{batch_counter:03d}]",
                        f"Train loss {l:.3f}",
                        f"Train acc {a:.3f}",
                        f"Throughput: {s:.2f} images/second",
                    )
                ),
            )

    return average_stats([losses, accuracies, world.size() * samples_per_sec], count)


def test_epoch(model, test_iter, epoch):
    accuracies = 0
    count = 0
    for batch_counter, batch in enumerate(test_iter):
        x = mx.array(batch["image"])
        y = mx.array(batch["label"])
        acc = eval_fn(model, x, y)
        accuracies += acc.item()
        count += 1

    with mx.stream(mx.cpu):
        accuracies = mx.distributed.all_sum(accuracies)
        count = mx.distributed.all_sum(count)
        return (accuracies / count).item()


def main(args):
    mx.random.seed(args.seed)

    # Initialize the distributed group and report the nodes that showed up
    world = mx.distributed.init()
    if world.size() > 1:
        print(f"Starting rank {world.rank()} of {world.size()}", flush=True)

    model = getattr(resnet, args.arch)()

    print_zero(world, f"Number of params: {model.num_params() / 1e6:0.04f} M")

    optimizer = optim.Adam(learning_rate=args.lr)

    train_data, test_data = get_cifar10(args.batch_size)
    for epoch in range(args.epochs):
        tr_loss, tr_acc, throughput = train_epoch(model, train_data, optimizer, epoch)
        print_zero(
            world,
            " | ".join(
                (
                    f"Epoch: {epoch}",
                    f"avg. Train loss {tr_loss:.3f}",
                    f"avg. Train acc {tr_acc:.3f}",
                    f"Throughput: {throughput:.2f} images/sec",
                )
            ),
        )

        test_acc = test_epoch(model, test_data, epoch)
        print_zero(world, f"Epoch: {epoch} | Test acc {test_acc:.3f}")

        train_data.reset()
        test_data.reset()


if __name__ == "__main__":
    args = parser.parse_args()
    if args.cpu:
        mx.set_default_device(mx.cpu)
    main(args)
