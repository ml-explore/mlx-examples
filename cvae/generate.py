import argparse

import mlx.core as mx
import model
import numpy as np
import utils


def generate(
    model_fname,
    num_latent_dims,
    num_img_channels,
    max_num_filters,
    outfile,
):
    # Load the model
    vae = model.CVAE(num_latent_dims, num_img_channels, max_num_filters)
    vae.load(model_fname)
    print(f"Loaded model with {num_latent_dims} latent dims from {model_fname}")

    # set model to eval mode
    vae.eval()

    # Generate a batch of random latent vectors.
    num_samples = 128

    # During training we have made sure that the distribution in latent
    # space remains close to a normal distribution, so we can sample
    # from a normal distribution to generate new images.
    z = mx.random.normal([num_samples, num_latent_dims])

    # Generate images from the latent vectors via the decoder
    images = vae.decode(z)

    # Save all images in a single file
    grid_image = utils.gen_grid_image_from_batch(images, num_rows=8)
    grid_image.save(outfile)
    print(f"Saved {num_samples} generated images to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU acceleration",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--model", type=str, required=True, help="Model filename *.pth")
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
        "--outfile",
        type=str,
        default="samples.png",
        help="Output filename for the generated samples, e.g. samples.png",
    )
    parser.add_argument(
        "--nimg_channels",
        type=int,
        default=1,
        help="Number of image channels (1 for grayscale, 3 for RGB)",
    )

    args = parser.parse_args()

    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    if args.cpu:
        mx.set_default_device(mx.cpu)

    utils.ensure_folder_exists(args.outfile)

    generate(
        args.model,
        args.latent_dims,
        args.nimg_channels,
        args.max_filters,
        args.outfile,
    )
