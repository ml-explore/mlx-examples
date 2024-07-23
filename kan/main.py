import argparse
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import mnist

from kan import KAN, KAN_Convolutional_Layer

from utils import print_trainable_parameters, save_model

def loss_fn(model, X, y):
    return mx.mean(nn.losses.cross_entropy(model(X), y))

def eval_fn(model, X, y):
    return mx.mean(mx.argmax(model(X), axis=1) == y)

def batch_iterate(batch_size, X, y):
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids], y[ids]

def main(args):
    seed = args.seed
    num_layers = args.num_layers
    hidden_dim = args.hidden_dim
    num_classes = args.num_classes
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    in_features = args.in_features
    out_features = args.out_features
    save_path = args.save_path
    eval_report_count = args.eval_report_count
    use_kan_convolution = args.use_kan_convolution

    np.random.seed(seed)

    print("Loading Dataset")
    train_images, train_labels, test_images, test_labels = map(
        mx.array, getattr(mnist, args.dataset)()
    )

    if use_kan_convolution:
        raise NotImplementedError("KAN Convolutional Layer functionality is not yet implemented.")
        # print("Loading and initializing KAN Convolutional Layer")
        # model = KAN_Convolutional_Layer(
        #     n_convs=num_layers,
        #     kernel_size=(in_features, out_features),
        #     hidden_act=nn.SiLU
        # )
    else:
        print("Loading and initializing KAN Fully Connected Layer")
        layers = [in_features * out_features] + [hidden_dim] * (num_layers - 1) + [num_classes]
        model = KAN(layers)

    mx.eval(model.parameters())
    print_trainable_parameters(model)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

    print(f"starting to train for {num_epochs} epochs")
    for e in range(num_epochs):
        tic = time.perf_counter()

        model.train()
        for X, y in batch_iterate(batch_size, train_images, train_labels):
            loss, grads = loss_and_grad_fn(model, X, y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

        toc = time.perf_counter()
        print(f"Epoch {e + 1}: Train Loss: {loss.item():.4f}, Time {toc - tic:.3f} (s)")

        if e == 1 or e % eval_report_count == 0 or e == num_epochs:
            tic = time.perf_counter()
            model.eval()
            accuracy = eval_fn(model, test_images, test_labels)
            toc = time.perf_counter()
            print(f"Epoch {e + 1}: Test accuracy {accuracy.item():.8f}, Time {toc - tic:.3f} (s)")

    save_model(model=model, save_path=save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train KAN on MNIST with MLX.")
    parser.add_argument("--cpu", action="store_true", help="Use the CPU instead og Metal GPU backend.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist"],
        help="The dataset to use.",
    )
    parser.add_argument("--use-kan-convolution", action="store_true", help="Use the Convolution KAN architecture.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of layers in the model.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Number of hidden units in each layer.")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of output classes.")
    parser.add_argument("--in-features", type=int, default=28, help="Number input features.")
    parser.add_argument("--out-features", type=int, default=28, help="Number output features.")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs to train.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for the optimizer.")
    parser.add_argument("--eval-report-count", type=int, default=10, help="Number of epochs to report validations / test accuracy.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--save-path", type=str, default="traned_kan_model", help="Random seed for reproducibility.")
    args = parser.parse_args()

    if args.cpu:
        mx.set_default_device(mx.cpu)

    main(args)