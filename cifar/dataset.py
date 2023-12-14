import mlx.core as mx
from mlx.data.datasets import load_cifar10
import math


def get_cifar10(batch_size, root=None):

    tr = load_cifar10(root=root)
    num_tr_samples = tr.size()

    mean = mx.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    std = mx.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

    tr_iter = (
        tr.shuffle()
        .to_stream()
        .image_random_h_flip("image", prob=0.5)
        .pad("image", 0, 4, 4, 0.0)
        .pad("image", 1, 4, 4, 0.0)
        .image_random_crop("image", 32, 32)
        .key_transform("image", lambda x: (x.astype("float32") / 255.0))
        .key_transform("image", lambda x: (x - mean) / std)
        .batch(batch_size)
    )

    test = load_cifar10(root=root, train=False)
    num_test_samples = test.size()

    test_iter = (
        test.to_stream()
        .key_transform("image", lambda x: (x.astype("float32") / 255.0))
        .key_transform("image", lambda x: (x - mean) / std)
        .batch(batch_size)
    )

    num_tr_steps_per_epoch = num_tr_samples // batch_size
    num_test_steps_per_epoch = num_test_samples // batch_size

    return tr_iter, test_iter
