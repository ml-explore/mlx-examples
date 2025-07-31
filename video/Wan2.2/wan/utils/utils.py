# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# utils MLX version
import argparse
import binascii
import logging
import os
import os.path as osp

import imageio
import mlx.core as mx
import numpy as np

__all__ = ['save_video', 'save_image', 'str2bool', 'masks_like', 'best_output_size']


def rand_name(length=8, suffix=''):
    name = binascii.b2a_hex(os.urandom(length)).decode('utf-8')
    if suffix:
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        name += suffix
    return name


def make_grid(tensor, nrow=8, normalize=True, value_range=(-1, 1)):
    """MLX equivalent of torchvision.utils.make_grid"""
    # tensor shape: (batch, channels, height, width)
    batch_size, channels, height, width = tensor.shape
    
    # Calculate grid dimensions
    ncol = nrow
    nrow_actual = (batch_size + ncol - 1) // ncol
    
    # Create grid
    grid_height = height * nrow_actual + (nrow_actual - 1) * 2  # 2 pixel padding
    grid_width = width * ncol + (ncol - 1) * 2
    
    # Initialize grid with zeros
    grid = mx.zeros((channels, grid_height, grid_width))
    
    # Fill grid
    for idx in range(batch_size):
        row = idx // ncol
        col = idx % ncol
        
        y_start = row * (height + 2)
        y_end = y_start + height
        x_start = col * (width + 2)
        x_end = x_start + width
        
        img = tensor[idx]
        if normalize:
            # Normalize to [0, 1]
            img = (img - value_range[0]) / (value_range[1] - value_range[0])
        
        grid[:, y_start:y_end, x_start:x_end] = img
    
    return grid


def save_video(tensor,
               save_file=None,
               fps=30,
               suffix='.mp4',
               nrow=8,
               normalize=True,
               value_range=(-1, 1)):
    # cache file
    cache_file = osp.join('/tmp', rand_name(
        suffix=suffix)) if save_file is None else save_file

    # save to cache
    try:
        # preprocess
        tensor = mx.clip(tensor, value_range[0], value_range[1])
        
        # tensor shape: (batch, channels, frames, height, width)
        # Process each frame
        frames = []
        for frame_idx in range(tensor.shape[2]):
            frame = tensor[:, :, frame_idx, :, :]  # (batch, channels, height, width)
            grid = make_grid(frame, nrow=nrow, normalize=normalize, value_range=value_range)
            frames.append(grid)
        
        # Stack frames and convert to (frames, height, width, channels)
        tensor = mx.stack(frames, axis=0)  # (frames, channels, height, width)
        tensor = mx.transpose(tensor, [0, 2, 3, 1])  # (frames, height, width, channels)
        
        # Convert to uint8
        tensor = (tensor * 255).astype(mx.uint8)
        tensor_np = np.array(tensor)

        # write video
        writer = imageio.get_writer(
            cache_file, fps=fps, codec='libx264', quality=8)
        for frame in tensor_np:
            writer.append_data(frame)
        writer.close()
    except Exception as e:
        logging.info(f'save_video failed, error: {e}')


def save_image(tensor, save_file, nrow=8, normalize=True, value_range=(-1, 1)):
    # cache file
    suffix = osp.splitext(save_file)[1]
    if suffix.lower() not in [
            '.jpg', '.jpeg', '.png', '.tiff', '.gif', '.webp'
    ]:
        suffix = '.png'

    # save to cache
    try:
        # Clip values
        tensor = mx.clip(tensor, value_range[0], value_range[1])
        
        # Make grid
        grid = make_grid(tensor, nrow=nrow, normalize=normalize, value_range=value_range)
        
        # Convert to (height, width, channels) and uint8
        grid = mx.transpose(grid, [1, 2, 0])  # (height, width, channels)
        grid = (grid * 255).astype(mx.uint8)
        
        # Save using imageio
        imageio.imwrite(save_file, np.array(grid))
        return save_file
    except Exception as e:
        logging.info(f'save_image failed, error: {e}')


def str2bool(v):
    """
    Convert a string to a boolean.

    Supported true values: 'yes', 'true', 't', 'y', '1'
    Supported false values: 'no', 'false', 'f', 'n', '0'

    Args:
        v (str): String to convert.

    Returns:
        bool: Converted boolean value.

    Raises:
        argparse.ArgumentTypeError: If the value cannot be converted to boolean.
    """
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v_lower in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected (True/False)')


def masks_like(tensor, zero=False, generator=None, p=0.2):
    """
    Generate masks similar to input tensors.
    
    Args:
        tensor: List of MLX arrays
        zero: Whether to apply zero masking
        generator: Random generator (for MLX, we use mx.random.seed instead)
        p: Probability for random masking
    
    Returns:
        Tuple of two lists of masks
    """
    assert isinstance(tensor, list)
    out1 = [mx.ones(u.shape, dtype=u.dtype) for u in tensor]
    out2 = [mx.ones(u.shape, dtype=u.dtype) for u in tensor]

    if zero:
        if generator is not None:
            # MLX doesn't have the same generator API as PyTorch
            # We'll use random state instead
            for u, v in zip(out1, out2):
                random_num = mx.random.uniform(0, 1, shape=(1,)).item()
                if random_num < p:
                    # Generate random values with normal distribution
                    normal_vals = mx.random.normal(shape=u[:, 0].shape, loc=-3.5, scale=0.5)
                    u[:, 0] = mx.exp(normal_vals)
                    v[:, 0] = mx.zeros_like(v[:, 0])
                else:
                    # Keep original values
                    u[:, 0] = u[:, 0]
                    v[:, 0] = v[:, 0]
        else:
            for u, v in zip(out1, out2):
                u[:, 0] = mx.zeros_like(u[:, 0])
                v[:, 0] = mx.zeros_like(v[:, 0])

    return out1, out2


def best_output_size(w, h, dw, dh, expected_area):
    """
    Calculate the best output size given constraints.
    
    Args:
        w: Width
        h: Height
        dw: Width divisor
        dh: Height divisor
        expected_area: Target area
    
    Returns:
        Tuple of (output_width, output_height)
    """
    # float output size
    ratio = w / h
    ow = (expected_area * ratio)**0.5
    oh = expected_area / ow

    # process width first
    ow1 = int(ow // dw * dw)
    oh1 = int(expected_area / ow1 // dh * dh)
    assert ow1 % dw == 0 and oh1 % dh == 0 and ow1 * oh1 <= expected_area
    ratio1 = ow1 / oh1

    # process height first
    oh2 = int(oh // dh * dh)
    ow2 = int(expected_area / oh2 // dw * dw)
    assert oh2 % dh == 0 and ow2 % dw == 0 and ow2 * oh2 <= expected_area
    ratio2 = ow2 / oh2

    # compare ratios
    if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2,
                                                 ratio2 / ratio):
        return ow1, oh1
    else:
        return ow2, oh2