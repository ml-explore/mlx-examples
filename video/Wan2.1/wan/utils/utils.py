import argparse
import binascii
import os
import os.path as osp

import imageio
import mlx.core as mx
import numpy as np

__all__ = ['cache_video', 'cache_image', 'str2bool']


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


def cache_video(tensor,
                save_file=None,
                fps=30,
                suffix='.mp4',
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
                retry=5):
    # cache file
    cache_file = osp.join('/tmp', rand_name(
        suffix=suffix)) if save_file is None else save_file

    # save to cache
    error = None
    for _ in range(retry):
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
            return cache_file
        except Exception as e:
            error = e
            continue
    else:
        print(f'cache_video failed, error: {error}', flush=True)
        return None


def save_image(tensor, save_file, nrow=8, normalize=True, value_range=(-1, 1)):
    """MLX equivalent of torchvision.utils.save_image"""
    # Make grid
    grid = make_grid(tensor, nrow=nrow, normalize=normalize, value_range=value_range)
    
    # Convert to (height, width, channels) and uint8
    grid = mx.transpose(grid, [1, 2, 0])  # (height, width, channels)
    grid = (grid * 255).astype(mx.uint8)
    
    # Save using imageio
    imageio.imwrite(save_file, np.array(grid))


def cache_image(tensor,
                save_file,
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
                retry=5):
    # cache file
    suffix = osp.splitext(save_file)[1]
    if suffix.lower() not in [
            '.jpg', '.jpeg', '.png', '.tiff', '.gif', '.webp'
    ]:
        suffix = '.png'

    # save to cache
    error = None
    for _ in range(retry):
        try:
            tensor = mx.clip(tensor, value_range[0], value_range[1])
            save_image(
                tensor,
                save_file,
                nrow=nrow,
                normalize=normalize,
                value_range=value_range)
            return save_file
        except Exception as e:
            error = e
            continue


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