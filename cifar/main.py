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


def eval_fn(model, inp, tgt):
    return mx.mean(mx.argmax(model(inp), axis=1) == tgt)


def train_epoch(model, train_iter, optimizer, epoch):
    def train_step(model, inp, tgt):
        output = model(inp)
        loss = mx.mean(nn.losses.cross_entropy(output, tgt))
        acc = mx.mean(mx.argmax(output, axis=1) == tgt)
        return loss, acc

    losses = []
    accs = []
    samples_per_sec = []

    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(inp, tgt):
        train_step_fn = nn.value_and_grad(model, train_step)
        (loss, acc), grads = train_step_fn(model, inp, tgt)
        optimizer.update(model, grads)
        return loss, acc

    for batch_counter, batch in enumerate(train_iter):
        x = mx.array(batch["image"])
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
        if batch_counter % 10 == 0:
            print(
                " | ".join(
                    (
                        f"Epoch {epoch:02d} [{batch_counter:03d}]",
                        f"Train loss {loss:.3f}",
                        f"Train acc {acc:.3f}",
                        f"Throughput: {throughput:.2f} images/second",
                    )
                )
            )

    mean_tr_loss = mx.mean(mx.array(losses))
    mean_tr_acc = mx.mean(mx.array(accs))
    samples_per_sec = mx.mean(mx.array(samples_per_sec))
    return mean_tr_loss, mean_tr_acc, samples_per_sec


def test_epoch(model, test_iter, epoch):
    accs = []
    for batch_counter, batch in enumerate(test_iter):
        x = mx.array(batch["image"])
        y = mx.array(batch["label"])
        acc = eval_fn(model, x, y)
        acc_value = acc.item()
        accs.append(acc_value)
    mean_acc = mx.mean(mx.array(accs))
    return mean_acc


def main(args):
    mx.random.seed(args.seed)

    model = getattr(resnet, args.arch)()

    print("Number of params: {:0.04f} M".format(model.num_params() / 1e6))

    optimizer = optim.Adam(learning_rate=args.lr)

    train_data, test_data = get_cifar10(args.batch_size)
    for epoch in range(args.epochs):
        tr_loss, tr_acc, throughput = train_epoch(model, train_data, optimizer, epoch)
        print(
            " | ".join(
                (
                    f"Epoch: {epoch}",
                    f"avg. Train loss {tr_loss.item():.3f}",
                    f"avg. Train acc {tr_acc.item():.3f}",
                    f"Throughput: {throughput.item():.2f} images/sec",
                )
            )
        )

        test_acc = test_epoch(model, test_data, epoch)
        print(f"Epoch: {epoch} | Test acc {test_acc.item():.3f}")

        train_data.reset()
        test_data.reset()


if __name__ == "__main__":
    args = parser.parse_args()
    if args.cpu:
        mx.set_default_device(mx.cpu)
    main(args)
