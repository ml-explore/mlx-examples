import mlx.core as mx
from mlx.data.datasets import load_cifar10
import math


def get_cifar10(batch_size, root=None):
    tr = load_cifar10(root=root)

    mean = mx.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    std = mx.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

    def normalize(x):
        x = x.astype("float32") / 255.0
        return (x - mean) / std

    tr_iter = (
        tr.shuffle()
        .to_stream()
        .image_random_h_flip("image", prob=0.5)
        .pad("image", 0, 4, 4, 0.0)
        .pad("image", 1, 4, 4, 0.0)
        .image_random_crop("image", 32, 32)
        .key_transform("image", normalize)
        .batch(batch_size)
    )

    test = load_cifar10(root=root, train=False)
    test_iter = test.to_stream().key_transform("image", normalize).batch(batch_size)

    return tr_iter, test_iter
