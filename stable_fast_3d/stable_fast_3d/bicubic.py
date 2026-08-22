"""
Exact reimplementation of PyTorch's bicubic interpolation (aten
UpSampleBicubic2d, cubic convolution with A=-0.75) in pure numpy.

MLX's nn.Upsample(mode="cubic") does NOT match PyTorch's kernel closely
enough (0.25 max abs diff on a synthetic test vs PyTorch F.interpolate,
while conv2d/gelu ports matched to ~1e-6). Since SF3D always calls
Dinov2Embeddings.interpolate_pos_encoding with the
SAME fixed 512x512 input (cond_image_size never varies), the interpolated
position embedding is IDENTICAL on every forward pass - just compute it
once at model-load time with this exact-match numpy port instead of
fighting MLX's kernel on every call.
"""

import numpy as np


def _cubic_convolution1(x, A):
    return ((A + 2) * x - (A + 3)) * x * x + 1


def _cubic_convolution2(x, A):
    return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A


def _cubic_coeffs(t, A=-0.75):
    x1 = t
    x2 = 1.0 - t
    return np.stack(
        [
            _cubic_convolution2(x1 + 1.0, A),
            _cubic_convolution1(x1, A),
            _cubic_convolution1(x2, A),
            _cubic_convolution2(x2 + 1.0, A),
        ],
        axis=-1,
    )  # (..., 4)


def bicubic_resize_nhwc(
    x: np.ndarray, out_h: int, out_w: int, scale_factor: float
) -> np.ndarray:
    """x: (B, H, W, C) float32. align_corners=False. Coordinate scale uses
    the LITERAL `scale_factor` passed to F.interpolate (1/scale_factor),
    NOT recomputed from the rounded (out_h, out_w) - PyTorch does not
    recompute here by default, confirmed empirically against a real
    F.interpolate(..., scale_factor=..., mode="bicubic") call (see
    debug_torch_ops.py / debug_mlx_ops.py): using in_h/out_h gave a 0.31
    max-abs-diff, while 1/scale_factor matched to 1.2e-6."""
    b, in_h, in_w, c = x.shape
    scale_h = 1.0 / scale_factor
    scale_w = 1.0 / scale_factor

    out_y = np.arange(out_h, dtype=np.float64)
    out_x = np.arange(out_w, dtype=np.float64)
    real_y = (out_y + 0.5) * scale_h - 0.5
    real_x = (out_x + 0.5) * scale_w - 0.5

    iy = np.floor(real_y).astype(np.int64)
    ix = np.floor(real_x).astype(np.int64)
    ty = (real_y - iy).astype(np.float32)
    tx = (real_x - ix).astype(np.float32)

    coeffs_y = _cubic_coeffs(ty)  # (out_h, 4)
    coeffs_x = _cubic_coeffs(tx)  # (out_w, 4)

    def clamp(idx, size):
        return np.clip(idx, 0, size - 1)

    y_idx = np.stack([clamp(iy + k - 1, in_h) for k in range(4)], axis=-1)  # (out_h,4)
    x_idx = np.stack([clamp(ix + k - 1, in_w) for k in range(4)], axis=-1)  # (out_w,4)

    # Gather rows: (B, out_h, 4, W, C)
    rows = x[:, y_idx, :, :]  # (B, out_h, 4, W, C)
    # Interpolate along x for each of the 4 gathered rows first (cheaper: gather cols too)
    # rows: (B, out_h, 4, W, C) -> gather x_idx along W axis
    cols = rows[:, :, :, x_idx, :]  # (B, out_h, 4, out_w, 4, C)
    # weighted sum over the last-gathered x taps (axis=4)
    out_along_x = np.einsum(
        "bhrwtc,wt->bhrwc", cols, coeffs_x
    )  # (B, out_h, 4, out_w, C)
    # weighted sum over the y taps (axis=2)
    out = np.einsum("bhrwc,hr->bhwc", out_along_x, coeffs_y)  # (B, out_h, out_w, C)
    return out.astype(np.float32)
