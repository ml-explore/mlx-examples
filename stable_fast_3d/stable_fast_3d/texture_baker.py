"""
texture_baker's rasterize/interpolate, ported from SF3D's
`texture_baker/texture_baker/csrc/baker.cpp` CPU reference
(rasterize_cpu/interpolate_cpu - the Metal kernel implements the identical
barycentric math, just BVH-accelerated on GPU).

No BVH here: instead of accelerating "which triangle contains this pixel"
with a tree, this bins each triangle to its pixel-space bounding box (small,
since ~30k triangles pack into a 512x512 atlas -> ~8px bbox on average) and
only barycentric-tests pixels in that box. First triangle to claim a pixel
wins (matches the reference's BVH.intersect returning on first hit; the UV
atlas is non-overlapping by construction so ties shouldn't occur in
practice). One-shot ~30k-triangle CPU op, not the inference bottleneck.
This module is plain numpy, no MLX needed.
"""

import numpy as np


def rasterize(uv: np.ndarray, indices: np.ndarray, bake_resolution: int) -> np.ndarray:
    """uv: (Nv,2) float32 in [0,1]. indices: (Nf,3) int. Returns (R,R,4)
    float32: [u,v,w barycentric, triangle_idx or -1]."""
    W = H = bake_resolution
    rast = np.zeros((H, W, 4), dtype=np.float32)
    rast[..., 3] = -1.0

    rows = np.arange(H, dtype=np.float32)
    cols = np.arange(W, dtype=np.float32)
    # Matches baker.cpp's rasterize_cpu exactly: idx = x*width+y (x=row),
    # pixel_coord = (y/height, 1 - x/width), no pixel-center offset.
    pcx_row = cols / H  # (W,) varies along columns (axis=1)
    pcy_col = 1.0 - rows / W  # (H,) varies along rows (axis=0)
    pcx = np.broadcast_to(pcx_row[None, :], (H, W))
    pcy = np.broadcast_to(pcy_col[:, None], (H, W))

    v0 = uv[indices[:, 0]].astype(np.float32)
    v1 = uv[indices[:, 1]].astype(np.float32)
    v2 = uv[indices[:, 2]].astype(np.float32)

    tri_min = np.minimum(np.minimum(v0, v1), v2)
    tri_max = np.maximum(np.maximum(v0, v1), v2)

    # Invert pixel_coord -> (row, col) index ranges per triangle bbox.
    col_lo = np.clip(np.floor(tri_min[:, 0] * H).astype(np.int64) - 1, 0, W - 1)
    col_hi = np.clip(np.ceil(tri_max[:, 0] * H).astype(np.int64) + 1, 0, W - 1)
    row_lo = np.clip(np.floor(W * (1.0 - tri_max[:, 1])).astype(np.int64) - 1, 0, H - 1)
    row_hi = np.clip(np.ceil(W * (1.0 - tri_min[:, 1])).astype(np.int64) + 1, 0, H - 1)

    v1v2 = v1 - v0
    v1v3 = v2 - v0
    d00 = (v1v2 * v1v2).sum(-1)
    d01 = (v1v2 * v1v3).sum(-1)
    d11 = (v1v3 * v1v3).sum(-1)
    denom = d00 * d11 - d01 * d01

    num_faces = indices.shape[0]
    for f in range(num_faces):
        r0, r1, c0, c1 = row_lo[f], row_hi[f], col_lo[f], col_hi[f]
        if r1 < r0 or c1 < c0:
            continue
        sub_pcx = pcx[r0 : r1 + 1, c0 : c1 + 1]
        sub_pcy = pcy[r0 : r1 + 1, c0 : c1 + 1]

        xyv1_x = sub_pcx - v0[f, 0]
        xyv1_y = sub_pcy - v0[f, 1]
        d20 = xyv1_x * v1v2[f, 0] + xyv1_y * v1v2[f, 1]
        d21 = xyv1_x * v1v3[f, 0] + xyv1_y * v1v3[f, 1]

        dn = denom[f]
        if dn == 0:
            continue
        vv = (d11[f] * d20 - d01[f] * d21) / dn
        ww = (d00[f] * d21 - d01[f] * d20) / dn
        uu = 1.0 - vv - ww

        hit = (vv >= 0.0) & (ww >= 0.0) & (vv + ww <= 1.0)
        sub_tri = rast[r0 : r1 + 1, c0 : c1 + 1, 3]
        write = hit & (sub_tri < 0)
        if not write.any():
            continue
        rast[r0 : r1 + 1, c0 : c1 + 1, 0][write] = uu[write]
        rast[r0 : r1 + 1, c0 : c1 + 1, 1][write] = vv[write]
        rast[r0 : r1 + 1, c0 : c1 + 1, 2][write] = ww[write]
        rast[r0 : r1 + 1, c0 : c1 + 1, 3][write] = f

    return rast


def get_mask(rast: np.ndarray) -> np.ndarray:
    return rast[..., -1] >= 0


def interpolate(attr: np.ndarray, rast: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """attr: (Nv,3). rast: (R,R,4). indices: (Nf,3). Returns (R,R,3)."""
    H, W = rast.shape[:2]
    tri_idx = rast[..., 3].astype(np.int64)
    bary = rast[..., :3]

    valid = tri_idx >= 0
    safe_idx = np.where(valid, tri_idx, 0)
    tri_verts = indices[safe_idx]  # (H,W,3) vertex indices per pixel

    v1 = attr[tri_verts[..., 0]]
    v2 = attr[tri_verts[..., 1]]
    v3 = attr[tri_verts[..., 2]]

    out = v1 * bary[..., 0:1] + v2 * bary[..., 1:2] + v3 * bary[..., 2:3]
    out = np.where(valid[..., None], out, 0.0).astype(np.float32)
    return out
