# Copyright © 2023 Apple Inc.

import gzip
import os
import pickle
from urllib import request

import numpy as np


def mnist(
    save_dir="/tmp", base_url="http://yann.lecun.com/exdb/mnist/", filename="mnist.pkl"
):
    """
    Load the MNIST dataset in 4 tensors: train images, train labels,
    test images, and test labels.

    Checks `save_dir` for already downloaded data otherwise downloads.

    Download code modified from:
      https://github.com/hsjeong5/MNIST-for-Numpy
    """

    def download_and_save(save_file):
        filename = [
            ["training_images", "train-images-idx3-ubyte.gz"],
            ["test_images", "t10k-images-idx3-ubyte.gz"],
            ["training_labels", "train-labels-idx1-ubyte.gz"],
            ["test_labels", "t10k-labels-idx1-ubyte.gz"],
        ]

        mnist = {}
        for name in filename:
            out_file = os.path.join("/tmp", name[1])
            request.urlretrieve(base_url + name[1], out_file)
        for name in filename[:2]:
            out_file = os.path.join("/tmp", name[1])
            with gzip.open(out_file, "rb") as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(
                    -1, 28 * 28
                )
        for name in filename[-2:]:
            out_file = os.path.join("/tmp", name[1])
            with gzip.open(out_file, "rb") as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
        with open(save_file, "wb") as f:
            pickle.dump(mnist, f)

    save_file = os.path.join(save_dir, filename)
    if not os.path.exists(save_file):
        download_and_save(save_file)
    with open(save_file, "rb") as f:
        mnist = pickle.load(f)

    def preproc(x):
        return x.astype(np.float32) / 255.0

    mnist["training_images"] = preproc(mnist["training_images"])
    mnist["test_images"] = preproc(mnist["test_images"])
    return (
        mnist["training_images"],
        mnist["training_labels"].astype(np.uint32),
        mnist["test_images"],
        mnist["test_labels"].astype(np.uint32),
    )


def fashion_mnist(save_dir="/tmp"):
    return mnist(
        save_dir,
        base_url="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
        filename="fashion_mnist.pkl",
    )


if __name__ == "__main__":
    train_x, train_y, test_x, test_y = mnist()
    assert train_x.shape == (60000, 28 * 28), "Wrong training set size"
    assert train_y.shape == (60000,), "Wrong training set size"
    assert test_x.shape == (10000, 28 * 28), "Wrong test set size"
    assert test_y.shape == (10000,), "Wrong test set size"
