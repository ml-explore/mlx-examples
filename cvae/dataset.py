import os
import tempfile

from mlx.data.datasets import load_mnist

import utils


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
    )

    # iterator over test set
    test_iter = (
        test.to_stream()
        .image_resize("image", h=img_size[0], w=img_size[1])
        .key_transform("image", normalize)
        .batch(batch_size)
    )
    return tr_iter, test_iter, num_img_channels


if __name__ == "__main__":
    batch_size = 32
    img_size = (64, 64)  # (H, W)

    tr_iter, test_iter, num_img_channels = mnist(
        batch_size=batch_size, img_size=img_size
    )

    B, H, W, C = batch_size, img_size[0], img_size[1], num_img_channels
    print(f"Batch size: {B}, Channels: {C}, Height: {H}, Width: {W}")

    batch_tr_iter = next(tr_iter)
    assert batch_tr_iter["image"].shape == (B, H, W, C), "Wrong training set size"
    assert batch_tr_iter["label"].shape == (batch_size,), "Wrong training set size"

    batch_test_iter = next(test_iter)
    assert batch_test_iter["image"].shape == (B, H, W, C), "Wrong training set size"
    assert batch_test_iter["label"].shape == (batch_size,), "Wrong training set size"

    # Save a batch as an image (as a sanity check)
    img = utils.gen_grid_image_from_batch(
        batch_tr_iter["image"], num_rows=4, norm_factor=255
    )

    temp_fname = os.path.join(tempfile.gettempdir(), "mnist_train_batch.png")
    img.save(temp_fname)
    print(f"Saved training batch to {temp_fname}")

    # Reset the iterators, if necessary
    tr_iter.reset()
    test_iter.reset()

    print("Dataset prepared successfully!")
