import math
import os

import mlx.core as mx
import numpy as np
from PIL import Image


def gen_grid_image_from_batch(image_batch, num_rows, norm_factor=255, dtype=np.uint8):
    """
    Generate a grid image from a batch of images. Assumes input of shape (B, H, W, C).
    """

    B, H, W, _ = image_batch.shape

    # Define the number of rows and columns in the grid
    num_cols = int(math.ceil(B / float(num_rows)))

    # Calculate the size of the output grid image
    grid_height = num_rows * H
    grid_width = num_cols * W

    # Normalize and convert to the desired data type
    image_batch = np.array(image_batch * norm_factor).astype(dtype)

    # Reshape the batch of images into a 2D grid
    grid_image = image_batch.reshape(num_rows, num_cols, H, W, -1)
    grid_image = grid_image.transpose(0, 2, 1, 3, 4)
    grid_image = grid_image.reshape(grid_height, grid_width, -1)

    # Convert the grid to a PIL Image
    return Image.fromarray(grid_image.squeeze())


def combine_and_save_image(img1_data, img2_data, fname):
    # Convert numpy arrays to Pillow images
    pimg1 = Image.fromarray(img1_data.squeeze())
    pimg2 = Image.fromarray(img2_data.squeeze())

    # Concatenate images horizontally
    total_width = pimg1.width + pimg2.width
    max_height = max(pimg1.height, pimg2.height)
    combined_img = Image.new("RGB", (total_width, max_height))

    # Paste the images side by side
    combined_img.paste(pimg1, (0, 0))
    combined_img.paste(pimg2, (pimg1.width, 0))

    # Save the combined image
    combined_img.save(fname)


def ensure_folder_exists(path):
    # Extract the directory path from the (file) path
    dir_path = os.path.dirname(path)

    # Check if the directory path is not empty
    if dir_path:
        # Check if the directory exists, and create it if it does not
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

    return dir_path  # Optionally return the directory path
