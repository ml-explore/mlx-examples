import argparse

import mlx.core as mx
import numpy as np

import dataset
import model
import utils


def reconstruct(
    model_fname,
    num_latent_dims,
    max_num_filters,
    rec_testdata,
    outdir,
):
    # Load the training and test data
    batch_size = 32

    # Image size
    img_size = (64, 64)

    # Get the data
    tr_iter, test_iter, num_img_channels = dataset.mnist(
        batch_size=batch_size, img_size=img_size
    )
    if rec_testdata:
        data_iter = test_iter
        suffix = "test"
    else:
        data_iter = tr_iter
        suffix = "train"

    # Load the model
    vae = model.CVAE(num_latent_dims, num_img_channels, max_num_filters)
    vae.load(model_fname)
    print(f"Loaded model with {num_latent_dims} latent dims from {model_fname}")

    # set model to eval mode
    vae.eval()

    # Loop over data and reconstruct
    img_count = 0
    img_path = f"./{outdir}/mnist_{suffix}/{num_latent_dims:04d}_latent_dims/img_"
    utils.ensure_folder_exists(img_path)

    for _, batch in enumerate(data_iter):
        images = mx.array(batch["image"])

        # reconstruct the images
        images_recon = vae(images)

        # save the images
        for j in range(images.shape[0]):
            img1 = images[j]
            img2 = images_recon[j]

            # Convert mlx arrays to numpy arrays and scale to 0-255
            img1_data = np.array(img1 * 255).astype(np.uint8)
            img2_data = np.array(img2 * 255).astype(np.uint8)

            # filename for image and reconstructed image
            img_fname = f"{img_path}{(img_count+j):08d}.png"
            utils.combine_and_save_image(img1_data, img2_data, img_fname)

        # print progress
        print(
            f"Reconstructed {img_count+images.shape[0]} images",
            end="\r",
            flush=True,
        )

        # update image count
        img_count = img_count + images.shape[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU (cuda/mps) acceleration",
    )
    parser.add_argument("--model", type=str, required=True, help="Model filename *.pth")
    parser.add_argument(
        "--rec_testdata",
        action="store_true",
        help="Reconstruct test split instead of training split",
    )

    parser.add_argument(
        "--latent_dims",
        type=int,
        required=True,
        help="Number of latent dimensions (positive integer)",
    )
    parser.add_argument(
        "--max_filters",
        type=int,
        default=64,
        help="Maximum number of filters in the convolutional layers",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory for the generated samples",
    )

    args = parser.parse_args()

    if args.cpu:
        mx.set_default_device(mx.cpu)

    if not args.rec_testdata:
        print("Reconstructing training data")
    else:
        print("Reconstructing test data")

    reconstruct(
        args.model,
        args.latent_dims,
        args.max_filters,
        args.rec_testdata,
        args.outdir,
    )
