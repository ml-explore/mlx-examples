"""
`query_triplane` (custom bilinear triplane sampler - MLX has no
F.grid_sample equivalent) + MaterialMLP (the 4-head decoder: density,
features/albedo, perturb_normal, vertex_offset). Ported from SF3D's
`sf3d/system.py::query_triplane` and `sf3d/models/network.py::MaterialMLP`.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

RADIUS = 0.87
N_NEURONS = 64
HEAD_HIDDEN_LAYERS = {
    "density": 2,
    "features": 3,
    "perturb_normal": 3,
    "vertex_offset": 2,
}
HEAD_OUT_CHANNELS = {
    "density": 1,
    "features": 3,
    "perturb_normal": 3,
    "vertex_offset": 3,
}
HEAD_OUT_BIAS = {
    "density": -1.0,
    "features": 0.0,
    "perturb_normal": 0.0,
    "vertex_offset": 0.0,
}


def grid_sample_bilinear(
    plane: mx.array, grid_x: mx.array, grid_y: mx.array
) -> mx.array:
    """Torch F.grid_sample(mode="bilinear", align_corners=True,
    padding_mode="zeros") equivalent for a SINGLE plane and a flat list of
    query points. plane: (C, H, W). grid_x, grid_y: (N,) in [-1, 1], where
    x indexes W and y indexes H (torch's grid[...,0]=x/width,
    grid[...,1]=y/height convention). Returns (C, N)."""
    c, h, w = plane.shape

    # align_corners=True: pixel = (grid + 1) / 2 * (size - 1)
    px = (grid_x + 1) * 0.5 * (w - 1)
    py = (grid_y + 1) * 0.5 * (h - 1)

    x0 = mx.floor(px)
    y0 = mx.floor(py)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = px - x0
    wy = py - y0

    def gather(xi, yi):
        # zero-padding for out-of-bounds, matching padding_mode="zeros"
        in_bounds = (xi >= 0) & (xi <= w - 1) & (yi >= 0) & (yi <= h - 1)
        xi_c = mx.clip(xi, 0, w - 1).astype(mx.int32)
        yi_c = mx.clip(yi, 0, h - 1).astype(mx.int32)
        vals = plane[:, yi_c, xi_c]  # (C, N)
        return vals * in_bounds.astype(plane.dtype)[None, :]

    v00 = gather(x0, y0)
    v01 = gather(x1, y0)
    v10 = gather(x0, y1)
    v11 = gather(x1, y1)

    top = v00 * (1 - wx)[None, :] + v01 * wx[None, :]
    bot = v10 * (1 - wx)[None, :] + v11 * wx[None, :]
    return top * (1 - wy)[None, :] + bot * wy[None, :]


def query_triplane(positions: mx.array, triplanes: mx.array) -> mx.array:
    """positions: (N, 3) world-space xyz. triplanes: (3, Cp, Hp, Wp) for a
    SINGLE batch element (matches how system.py calls this per-mesh with
    scene_codes[i], and how the reference was captured with batch size 1
    via the unbatched code path). Returns (N, 3*Cp)."""
    positions = positions * (1.0 / RADIUS)  # scale_tensor((-radius,radius)->(-1,1))

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    # planes 0,1,2 sample axis pairs [0,1], [0,2], [1,2] - first component
    # is grid's x (width), second is grid's y (height), per system.py's
    # indices2D stacking + grid_sample's (x,y) convention.
    axis_pairs = [(x, y), (x, z), (y, z)]

    feats = []
    for p in range(3):
        gx, gy = axis_pairs[p]
        feats.append(grid_sample_bilinear(triplanes[p], gx, gy))  # (Cp, N)
    out = mx.concatenate(feats, axis=0)  # (3*Cp, N)
    return out.transpose(1, 0)  # (N, 3*Cp)


def trunc_exp(x: mx.array) -> mx.array:
    # torch-ngp's trunc_exp (network.py's _TruncExp) only truncates the
    # BACKWARD gradient (backward() clamps x to max=15 before the exp used
    # in the gradient) - the forward pass is plain, unclamped torch.exp(x).
    # No truncation needed for inference.
    return mx.exp(x)


class MLPHead(nn.Module):
    def __init__(self, name: str, in_channels: int = 120):
        super().__init__()
        n_hidden = HEAD_HIDDEN_LAYERS[name]
        out_ch = HEAD_OUT_CHANNELS[name]
        dims = [in_channels] + [N_NEURONS] * n_hidden + [out_ch]
        self.linears = [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]

    def __call__(self, x: mx.array) -> mx.array:
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i != len(self.linears) - 1:
                x = nn.silu(x)
        return x


class MaterialMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = {name: MLPHead(name) for name in HEAD_HIDDEN_LAYERS}

    def __call__(self, x: mx.array, names) -> dict:
        out = {}
        for name in names:
            raw = self.heads[name](x) + HEAD_OUT_BIAS[name]
            if name == "density":
                out[name] = trunc_exp(raw)
            elif name == "features":
                out[name] = mx.sigmoid(raw)
            elif name == "perturb_normal":
                norm = mx.clip(mx.linalg.norm(raw, axis=-1, keepdims=True), 1e-7, None)
                out[name] = raw / norm
            elif name == "vertex_offset":
                out[name] = raw
        return out


def load_into(model: MaterialMLP, weights: dict, prefix: str = "decoder."):
    for name, head in model.heads.items():
        n_hidden = HEAD_HIDDEN_LAYERS[name]
        # Sequential indices: Linear at 0,2,4,... (SiLU at odd indices has no params)
        indices = [2 * i for i in range(n_hidden + 1)]
        for lin, idx in zip(head.linears, indices):
            lin.weight = weights[f"{prefix}heads.{name}.{idx}.weight"]
            lin.bias = weights[f"{prefix}heads.{name}.{idx}.bias"]
