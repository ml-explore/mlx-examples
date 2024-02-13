# Copyright © 2023-2024 Apple Inc.

from mlx.data.datasets import load_mnist


def mnist(batch_size, img_size, root=None):
    # load train and test sets using mlx-data
    load_fn = load_mnist
    tr = load_fn(root=root, train=True)
    test = load_fn(root=root, train=False)

    # number of image channels is 1 for MNIST
    num_img_channels = 1

    # normalize to [0,1]
    def normalize(x):
        return x.astype("float32") / 255.0

    # iterator over training set
    tr_iter = (
        tr.shuffle()
        .to_stream()
        .image_resize("image", h=img_size[0], w=img_size[1])
        .key_transform("image", normalize)
        .batch(batch_size)
        .prefetch(4, 4)
    )

    # iterator over test set
    test_iter = (
        test.to_stream()
        .image_resize("image", h=img_size[0], w=img_size[1])
        .key_transform("image", normalize)
        .batch(batch_size)
    )
    return tr_iter, test_iter


if __name__ == "__main__":
    batch_size = 32
    img_size = (64, 64)  # (H, W)

    tr_iter, test_iter = mnist(batch_size=batch_size, img_size=img_size)

    B, H, W, C = batch_size, img_size[0], img_size[1], 1
    print(f"Batch size: {B}, Channels: {C}, Height: {H}, Width: {W}")

    batch_tr_iter = next(tr_iter)
    assert batch_tr_iter["image"].shape == (B, H, W, C), "Wrong training set size"
    assert batch_tr_iter["label"].shape == (batch_size,), "Wrong training set size"

    batch_test_iter = next(test_iter)
    assert batch_test_iter["image"].shape == (B, H, W, C), "Wrong training set size"
    assert batch_test_iter["label"].shape == (batch_size,), "Wrong training set size"
