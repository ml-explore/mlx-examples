import argparse
import time
import resnet
import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as optim
from dataset import get_cifar10


parser = argparse.ArgumentParser(add_help=True)
parser.add_argument(
    "--arch",
    type=str,
    default="resnet20",
    help="model architecture [resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202]",
)
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--cpu", action="store_true", help="use cpu only")


def loss_fn(model, inp, tgt):
    return mx.mean(nn.losses.cross_entropy(model(inp), tgt))


def eval_fn(model, inp, tgt):
    return mx.mean(mx.argmax(model(inp), axis=1) == tgt)


def train_epoch(model, train_iter, optimizer, epoch):
    def train_step(model, inp, tgt):
        output = model(inp)
        loss = mx.mean(nn.losses.cross_entropy(output, tgt))
        acc = mx.mean(mx.argmax(output, axis=1) == tgt)
        return loss, acc

    train_step_fn = nn.value_and_grad(model, train_step)

    losses = []
    accs = []
    samples_per_sec = []

    for batch_counter, batch in enumerate(train_iter):
        x = mx.array(batch["image"])
        y = mx.array(batch["label"])
        tic = time.perf_counter()
        (loss, acc), grads = train_step_fn(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        toc = time.perf_counter()
        loss_value = loss.item()
        acc_value = acc.item()
        losses.append(loss_value)
        accs.append(acc_value)
        samples_per_sec.append(x.shape[0] / (toc - tic))
        if batch_counter % 10 == 0:
            print(
                f"Epoch {epoch:02d} [{batch_counter:03d}] | tr_loss {loss_value:.3f} | tr_acc {acc_value:.3f} | Throughput: {x.shape[0] / (toc - tic):.2f} images/second"
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

    model = resnet.__dict__[args.arch]()

    print("num_params: {:0.04f} M".format(model.num_params() / 1e6))
    mx.eval(model.parameters())

    optimizer = optim.Adam(learning_rate=args.lr)

    train_data, test_data = get_cifar10(args.batch_size)
    for epoch in range(args.epochs):
        epoch_tr_loss, epoch_tr_acc, train_throughput = train_epoch(
            model, train_data, optimizer, epoch
        )
        print(
            f"Epoch: {epoch} | avg. tr_loss {epoch_tr_loss.item():.3f} | avg. tr_acc {epoch_tr_acc.item():.3f} | Train Throughput: {train_throughput.item():.2f} images/sec"
        )

        epoch_test_acc = test_epoch(model, test_data, epoch)
        print(f"Epoch: {epoch} | test_acc {epoch_test_acc.item():.3f}")

        train_data.reset()
        test_data.reset()


if __name__ == "__main__":
    args = parser.parse_args()
    if args.cpu:
        mx.set_default_device(mx.cpu)
    main(args)
