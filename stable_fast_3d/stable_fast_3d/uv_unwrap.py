"""
UV unwrapping, ported from SF3D's `uv_unwrapper/uv_unwrapper/unwrap.py` (a
box/cube-map projection unwrapper, NOT xatlas-style - almost entirely plain
tensor math, ported 1:1 to numpy) plus `sf3d/models/mesh.py`'s
Mesh.unwrap_uv/_compute_vertex_normal/_compute_vertex_tangent (the
per-corner flattening + normal/tangent recompute that happens right after).

The one non-trivial piece is uv_unwrapper's single compiled op,
`torch.ops.UVUnwrapper.assign_faces_uv_to_atlas_index` (csrc/unwrapper.cpp):
a per-cube-face BVH over the 2D UV-projected triangles that detects
overlapping ("occluded") triangles and moves them to a duplicate atlas
slot. It's CPU-only (OpenMP, no CUDA/Metal) - a BVH would give nothing on
MLX's GPU-array model, so it's reimplemented here as a grid-hashed
triangle-triangle overlap test (bin triangles into a coarse UV grid, only
test pairs sharing a bin) rather than porting the BVH itself.

torch.pca_lowrank's random seed (torch.manual_seed(0)) is NOT reproduced -
it uses a randomized approximate SVD; this port uses a plain deterministic
np.linalg.svd on the (tiny, Nx3) centered point cloud instead, which is
exact rather than approximate. The two axes can come out sign-flipped
relative to torch's (PCA sign ambiguity), which changes the atlas's
absolute orientation but not its validity (non-overlapping unwrap).
This module is plain numpy, no MLX needed.
"""

from typing import Tuple

import numpy as np


def _normalize(x: np.ndarray, axis=-1, eps=1e-6) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, eps, None)


def _align_mesh_with_main_axis(
    vertex_positions: np.ndarray, vertex_normals: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    centered = vertex_positions - vertex_positions.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    main_axis, secondary_axis = vt[0], vt[1]

    main_axis = _normalize(main_axis, axis=-1)
    secondary_axis = (
        secondary_axis - (secondary_axis * main_axis).sum(-1, keepdims=True) * main_axis
    )
    secondary_axis = _normalize(secondary_axis, axis=-1)
    third_axis = _normalize(np.cross(main_axis, secondary_axis), axis=-1)

    main_idx = int(np.argmax(np.abs(main_axis)))
    sec_idx = int(np.argmax(np.abs(secondary_axis)))
    third_idx = int(np.argmax(np.abs(third_axis)))

    all_axes = {0, 1, 2}
    cur_index = 1
    while len({main_idx, sec_idx, third_idx}) != 3:
        missing = (all_axes - {main_idx, sec_idx, third_idx}).pop()
        if cur_index == 1:
            third_idx = missing
        elif cur_index == 2:
            sec_idx = missing
        else:
            raise ValueError("Could not find 3 unique axis")
        cur_index += 1

    axes = [None, None, None]
    axes[main_idx] = main_axis
    axes[sec_idx] = secondary_axis
    axes[third_idx] = third_axis
    rot_mat = np.stack(axes, axis=1).T  # (3,3)

    new_pos = vertex_positions @ rot_mat.T
    new_nrm = vertex_normals @ rot_mat.T
    return new_pos, new_nrm


def _box_assign_vertex_to_cube_face(
    vertex_positions: np.ndarray,
    vertex_normals: np.ndarray,
    triangle_idxs: np.ndarray,
    bbox: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    v_pos_normalized = (vertex_positions - bbox[:1]) / (bbox[1:] - bbox[:1])
    v_pos_normalized = 2.0 * v_pos_normalized - 1.0

    v0 = v_pos_normalized[triangle_idxs[:, 0]]
    v1 = v_pos_normalized[triangle_idxs[:, 1]]
    v2 = v_pos_normalized[triangle_idxs[:, 2]]
    tri_stack = np.stack([v0, v1, v2], axis=1)  # (Nf,3,3)

    vn0 = vertex_normals[triangle_idxs[:, 0]]
    vn1 = vertex_normals[triangle_idxs[:, 1]]
    vn2 = vertex_normals[triangle_idxs[:, 2]]
    tri_stack_nrm = np.stack([vn0, vn1, vn2], axis=1)

    face_normal = _normalize(tri_stack_nrm.sum(1), axis=-1)

    abs_x, abs_y, abs_z = (
        np.abs(tri_stack[..., 0]),
        np.abs(tri_stack[..., 1]),
        np.abs(tri_stack[..., 2]),
    )

    axis = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=face_normal.dtype,
    )
    face_normal_axis = (face_normal[:, None, :] * axis[None, :, :]).sum(-1)
    index = face_normal_axis.argmax(-1)  # (Nf,)

    max_axis = np.ones_like(abs_x)
    uc = np.zeros(tri_stack.shape[:2] + (1,), dtype=tri_stack.dtype)
    vc = np.zeros(tri_stack.shape[:2] + (1,), dtype=tri_stack.dtype)

    def set_face(mask, u_src, v_src, a_src):
        max_axis[mask] = a_src[mask]
        uc[mask] = u_src[mask][..., None]
        vc[mask] = v_src[mask][..., None]

    set_face(index == 0, tri_stack[..., 1], -tri_stack[..., 2], abs_x)
    set_face(index == 1, tri_stack[..., 1], -tri_stack[..., 2], abs_x)
    set_face(index == 2, tri_stack[..., 0], -tri_stack[..., 2], abs_y)
    set_face(index == 3, tri_stack[..., 0], -tri_stack[..., 2], abs_y)
    set_face(index == 4, tri_stack[..., 0], tri_stack[..., 1], abs_z)
    set_face(index == 5, tri_stack[..., 0], -tri_stack[..., 1], abs_z)

    max_dim_div = max_axis.max(axis=0, keepdims=True)
    uc = np.clip((uc[..., 0] / max_dim_div + 1.0) * 0.5, 0, 1)
    vc = np.clip((vc[..., 0] / max_dim_div + 1.0) * 0.5, 0, 1)

    uv = np.stack([uc, vc], axis=-1)  # (Nf,3,2)
    return uv, index


def _calculate_tangents(
    vertex_positions: np.ndarray,
    vertex_normals: np.ndarray,
    triangle_idxs: np.ndarray,
    face_uv: np.ndarray,
) -> np.ndarray:
    pos = [vertex_positions[triangle_idxs[:, i]] for i in range(3)]
    vn_idx = [triangle_idxs[:, i] for i in range(3)]
    tex = [face_uv[:, i] for i in range(3)]

    tangents = np.zeros_like(vertex_normals)
    tansum = np.zeros_like(vertex_normals)

    duv1 = tex[1] - tex[0]
    duv2 = tex[2] - tex[0]
    dpos1 = pos[1] - pos[0]
    dpos2 = pos[2] - pos[0]

    tng_nom = dpos1 * duv2[..., 1:2] - dpos2 * duv1[..., 1:2]
    denom = duv1[..., 0:1] * duv2[..., 1:2] - duv1[..., 1:2] * duv2[..., 0:1]
    denom_safe = np.clip(denom, 1e-6, None)
    tang = tng_nom / denom_safe

    for i in range(3):
        idx = vn_idx[i]
        np.add.at(tangents, idx, tang)
        np.add.at(tansum, idx, np.ones_like(tang))

    tangents = tangents / tansum
    tangents = _normalize(tangents, axis=1)
    tangents = _normalize(
        tangents - (tangents * vertex_normals).sum(-1, keepdims=True) * vertex_normals,
        axis=1,
    )
    return tangents


def _rotate_uv_slices_consistent_space(
    vertex_positions: np.ndarray,
    vertex_normals: np.ndarray,
    triangle_idxs: np.ndarray,
    uv: np.ndarray,
    index: np.ndarray,
) -> np.ndarray:
    tangents = _calculate_tangents(vertex_positions, vertex_normals, triangle_idxs, uv)
    pos_stack = np.stack(
        [
            -vertex_positions[..., 1],
            vertex_positions[..., 0],
            np.zeros_like(vertex_positions[..., 0]),
        ],
        axis=-1,
    )
    expected_tangents = _normalize(
        np.cross(vertex_normals, np.cross(pos_stack, vertex_normals, axis=-1), axis=-1),
        axis=-1,
    )

    actual_tangents = tangents[triangle_idxs]  # (Nf,3,3)
    expected_tangents = expected_tangents[triangle_idxs]

    index_mod = index % 6
    for i in range(6):
        mask = index_mod == i
        if not mask.any():
            continue
        actual_mean = actual_tangents[mask].mean(axis=(0, 1))
        expected_mean = expected_tangents[mask].mean(axis=(0, 1))

        dot = float(np.dot(actual_mean, expected_mean))
        cross = float(
            actual_mean[0] * expected_mean[1] - actual_mean[1] * expected_mean[0]
        )
        angle = np.arctan2(cross, dot)

        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]], dtype=uv.dtype)

        uv_cur = uv[mask] * 2 - 1
        rotated = np.einsum("ij,nfj->nfi", rot, uv_cur)
        uv[mask] = rotated
        uv[mask] = (uv[mask] - uv[mask].min()) / (uv[mask].max() - uv[mask].min())

    return uv


def _assign_faces_uv_to_atlas_index(
    vertex_positions: np.ndarray,
    triangle_idxs: np.ndarray,
    face_uv: np.ndarray,
    face_index: np.ndarray,
) -> np.ndarray:
    """Reimplements uv_unwrapper's BVH-based overlap check (csrc/
    unwrapper.cpp) as a grid-hashed 2D triangle overlap test: bin each
    cube-face group's triangles into a coarse grid by AABB, only test pairs
    sharing a cell, move losers of an overlap to index+6 (clamped to 12).
    The overlap tiebreak (which of the pair is "in front") compares the 3D
    vertex-position centroids along the cube face's axis, matching
    unwrapper.cpp's vertex_tri_centroids (NOT the 2D UV centroid)."""
    num_faces = triangle_idxs.shape[0]
    assign_indices = face_index.copy()

    tri_3d = vertex_positions[triangle_idxs]  # (Nf,3,3)
    centroids_3d = tri_3d.mean(axis=1)  # (Nf,3) world-space centroid
    tri_min = face_uv.min(axis=1)
    tri_max = face_uv.max(axis=1)

    for offset in (0, 6):
        # groups: offset..offset+5 (6 cube faces at this overlap level)
        for face_i in range(offset, offset + 6):
            members = np.nonzero(assign_indices == face_i)[0]
            if members.shape[0] < 2:
                continue

            GRID = 24
            lo = tri_min[members].min(0)
            hi = tri_max[members].max(0)
            span = np.clip(hi - lo, 1e-9, None)

            cell_lo = np.clip(
                ((tri_min[members] - lo) / span * GRID).astype(np.int64), 0, GRID - 1
            )
            cell_hi = np.clip(
                ((tri_max[members] - lo) / span * GRID).astype(np.int64), 0, GRID - 1
            )

            from collections import defaultdict

            bins = defaultdict(list)
            for local_i, m in enumerate(members):
                for cx in range(cell_lo[local_i, 0], cell_hi[local_i, 0] + 1):
                    for cy in range(cell_lo[local_i, 1], cell_hi[local_i, 1] + 1):
                        bins[(cx, cy)].append(local_i)

            candidate_pairs = set()
            for cell_members in bins.values():
                if len(cell_members) < 2:
                    continue
                for a in range(len(cell_members)):
                    for b in range(a + 1, len(cell_members)):
                        i, j = cell_members[a], cell_members[b]
                        candidate_pairs.add((i, j) if i < j else (j, i))

            axis = 0 if (face_i % 6) < 2 else (1 if (face_i % 6) < 4 else 2)
            use_max = (face_i % 2) == 1
            occluded = set()
            for i, j in candidate_pairs:
                gi, gj = members[i], members[j]
                if gi in occluded or gj in occluded:
                    continue
                if not _triangles_overlap_2d(face_uv[gi], face_uv[gj]):
                    continue
                pa = centroids_3d[gi, axis]
                pb = centroids_3d[gj, axis]
                first, second = (gi, gj)
                if use_max:
                    if pa < pb:
                        first, second = gj, gi
                else:
                    if pa > pb:
                        first, second = gj, gi
                occluded.add(second)

            for occ in occluded:
                new_idx = min(assign_indices[occ] + 6, 12)
                assign_indices[occ] = new_idx

    return assign_indices


_OVERLAP_EPS = 1e-5


def _triangles_overlap_2d(tri_a: np.ndarray, tri_b: np.ndarray) -> bool:
    """Separating Axis Theorem test for two 2D triangles (3x2 arrays). Uses
    an epsilon slack on the separating condition so triangles that only
    touch (shared edge/vertex between mesh neighbors - the overwhelmingly
    common case, since every triangle borders 3 others) are NOT counted as
    overlapping - only genuine area intersection is. Mirrors intersect.cpp's
    EPSILON-guarded coplanar tests (it never treats exact touching as a
    real intersection either)."""

    def edges(tri):
        return [tri[(i + 1) % 3] - tri[i] for i in range(3)]

    for tri in (tri_a, tri_b):
        for e in edges(tri):
            axis = np.array([-e[1], e[0]])
            axis_len = np.linalg.norm(axis)
            if axis_len < 1e-12:
                continue
            axis = axis / axis_len
            proj_a = tri_a @ axis
            proj_b = tri_b @ axis
            if (
                proj_a.max() <= proj_b.min() + _OVERLAP_EPS
                or proj_b.max() <= proj_a.min() + _OVERLAP_EPS
            ):
                return False
    return True


def _find_slice_offset_and_scale(index: np.ndarray):
    off = 1 / 3
    dupl_off = 1 / 6

    def x_offset_calc(x, i):
        oc = i // 6
        return off * x if oc == 0 else dupl_off * x + min(oc - 1, 1) * 0.5

    def y_offset_calc(x, i):
        oc = i // 6
        return off * x if oc == 0 else dupl_off * x + off * 2

    offset_x = np.zeros_like(index, dtype=np.float64)
    offset_y = np.zeros_like(index, dtype=np.float64)
    offset_x_vals = [0, 1, 2, 0, 1, 2]
    offset_y_vals = [0, 0, 0, 1, 1, 1]

    for i in range(int(index.max()) + 1):
        mask = index == i
        if not mask.any():
            continue
        offset_x[mask] = x_offset_calc(offset_x_vals[i % 6], i)
        offset_y[mask] = y_offset_calc(offset_y_vals[i % 6], i)

    div_x = np.full_like(index, 3, dtype=np.float64)
    div_x[index >= 6] = 6
    div_y = div_x.copy()
    div_x[index >= 12] = 2
    div_y[index >= 12] = 3

    return offset_x, offset_y, div_x, div_y


def _handle_slice_uvs(
    uv: np.ndarray, index: np.ndarray, island_padding: float, max_index: int = 12
) -> np.ndarray:
    uc, vc = uv[..., 0].copy(), uv[..., 1].copy()

    for i in range(6, max_index):
        fi = index == i
        if fi.sum() == 0:
            continue
        uc[fi] = (uc[fi] - uc[fi].min()) / max(uc[fi].max() - uc[fi].min(), 0.5)
        vc[fi] = (vc[fi] - vc[fi].min()) / max(vc[fi].max() - vc[fi].min(), 0.5)

    uc_p = np.clip(uc * (1 - 2 * island_padding) + island_padding, 0, 1)
    vc_p = np.clip(vc * (1 - 2 * island_padding) + island_padding, 0, 1)
    return np.stack([uc_p, vc_p], axis=-1)


def _handle_remaining_uvs(
    uv: np.ndarray, index: np.ndarray, island_padding: float
) -> np.ndarray:
    import math

    remaining = index >= 12
    squares_left = int(remaining.sum())
    if squares_left == 0:
        return uv

    uc = uv[remaining][..., 0]
    vc = uv[remaining][..., 1]

    ratio = 0.5 * (1 / 3)
    mult = math.sqrt(squares_left / ratio)
    num_w = int(math.ceil(0.5 * mult))
    num_h = int(math.ceil(squares_left / max(num_w, 1)))
    num_w = max(num_w, 1)
    num_h = max(num_h, 1)

    width = 1 / num_w
    height = 1 / num_h
    clip_val = min(width, height) * 1.5

    uc_min = uc.min(axis=1, keepdims=True)
    uc_max = uc.max(axis=1, keepdims=True)
    vc_min = vc.min(axis=1, keepdims=True)
    vc_max = vc.max(axis=1, keepdims=True)

    uc = (uc - uc_min) / np.clip(uc_max - uc_min, clip_val, None)
    vc = (vc - vc_min) / np.clip(vc_max - vc_min, clip_val, None)

    uc = np.clip(
        uc * (1 - island_padding * num_w * 0.5) + island_padding * num_w * 0.25, 0, 1
    )
    vc = np.clip(
        vc * (1 - island_padding * num_h * 0.5) + island_padding * num_h * 0.25, 0, 1
    )

    uc = uc * width
    vc = vc * height

    idx = np.arange(uc.shape[0])
    x_idx = idx % num_w
    y_idx = idx // num_w
    uc = uc + x_idx[:, None] * width
    vc = vc + y_idx[:, None] * height

    uc = np.clip(uc * (1 - 2 * island_padding * 0.5) + island_padding * 0.5, 0, 1)
    vc = np.clip(vc * (1 - 2 * island_padding * 0.5) + island_padding * 0.5, 0, 1)

    uv = uv.copy()
    uv[remaining] = np.stack([uc, vc], axis=-1)
    return uv


def _distribute_individual_uvs_in_atlas(
    face_uv, assigned_faces, offset_x, offset_y, div_x, div_y, island_padding
):
    placed_uv = _handle_slice_uvs(face_uv, assigned_faces, island_padding)
    placed_uv = _handle_remaining_uvs(placed_uv, assigned_faces, island_padding)

    uc, vc = placed_uv[..., 0], placed_uv[..., 1]
    uc = uc / div_x[:, None] + offset_x[:, None]
    vc = vc / div_y[:, None] + offset_y[:, None]

    return np.stack([uc, vc], axis=-1).reshape(-1, 2)


def unwrap_mesh(
    vertex_positions: np.ndarray,
    vertex_normals: np.ndarray,
    triangle_idxs: np.ndarray,
    island_padding: float = 0.02,
) -> np.ndarray:
    """Verbatim (module-structure-wise) port of uv_unwrapper.Unwrapper.forward.
    Returns flat per-face-corner UVs (Nf*3, 2) - the atlas index dedup step
    (_get_unique_face_uv) is skipped since sf3d's Mesh.unwrap_uv immediately
    re-flattens to per-corner anyway (see module docstring)."""
    vertex_positions, vertex_normals = _align_mesh_with_main_axis(
        vertex_positions, vertex_normals
    )
    bbox = np.stack([vertex_positions.min(0), vertex_positions.max(0)], axis=0)

    face_uv, face_index = _box_assign_vertex_to_cube_face(
        vertex_positions, vertex_normals, triangle_idxs, bbox
    )
    face_uv = _rotate_uv_slices_consistent_space(
        vertex_positions, vertex_normals, triangle_idxs, face_uv, face_index
    )
    assigned = _assign_faces_uv_to_atlas_index(
        vertex_positions, triangle_idxs, face_uv, face_index
    )
    offset_x, offset_y, div_x, div_y = _find_slice_offset_and_scale(assigned)
    placed_uv = _distribute_individual_uvs_in_atlas(
        face_uv, assigned, offset_x, offset_y, div_x, div_y, island_padding
    )
    return placed_uv  # (Nf*3, 2)


def compute_vertex_normal(v_pos: np.ndarray, t_pos_idx: np.ndarray) -> np.ndarray:
    i0, i1, i2 = t_pos_idx[:, 0], t_pos_idx[:, 1], t_pos_idx[:, 2]
    v0, v1, v2 = v_pos[i0], v_pos[i1], v_pos[i2]
    face_normals = np.cross(v1 - v0, v2 - v0)

    v_nrm = np.zeros_like(v_pos)
    np.add.at(v_nrm, i0, face_normals)
    np.add.at(v_nrm, i1, face_normals)
    np.add.at(v_nrm, i2, face_normals)

    degenerate = (v_nrm * v_nrm).sum(-1) <= 1e-20
    v_nrm[degenerate] = np.array([0.0, 0.0, 1.0])
    return _normalize(v_nrm, axis=1)


def compute_vertex_tangent(
    v_pos: np.ndarray, v_tex: np.ndarray, v_nrm: np.ndarray, t_pos_idx: np.ndarray
) -> np.ndarray:
    pos = [v_pos[t_pos_idx[:, i]] for i in range(3)]
    tex = [v_tex[t_pos_idx[:, i]] for i in range(3)]
    vn_idx = [t_pos_idx[:, i] for i in range(3)]

    tangents = np.zeros_like(v_nrm)
    tansum = np.zeros_like(v_nrm)

    duv1 = tex[1] - tex[0]
    duv2 = tex[2] - tex[0]
    dpos1 = pos[1] - pos[0]
    dpos2 = pos[2] - pos[0]

    tng_nom = dpos1 * duv2[..., 1:2] - dpos2 * duv1[..., 1:2]
    denom = duv1[..., 0:1] * duv2[..., 1:2] - duv1[..., 1:2] * duv2[..., 0:1]
    tang = tng_nom / np.clip(denom, 1e-6, None)

    for i in range(3):
        idx = vn_idx[i]
        np.add.at(tangents, idx, tang)
        np.add.at(tansum, idx, np.ones_like(tang))

    tangents = tangents / tansum
    tangents = _normalize(tangents, axis=1)
    tangents = _normalize(
        tangents - (tangents * v_nrm).sum(-1, keepdims=True) * v_nrm, axis=1
    )
    return tangents


def mesh_unwrap_uv(
    v_pos: np.ndarray, t_pos_idx: np.ndarray, island_padding: float = 0.02
):
    """Full port of Mesh.unwrap_uv: unwraps + flattens to per-face-corner
    representation + recomputes normals/tangents on the flattened mesh.
    Returns (v_pos_flat (Nf*3,3), t_pos_idx_flat (Nf*3,3) trivial 0,1,2.. ,
    v_tex_flat (Nf*3,2), v_nrm_flat (Nf*3,3), v_tng_flat (Nf*3,3))."""
    v_nrm = compute_vertex_normal(v_pos, t_pos_idx)
    uv_flat = unwrap_mesh(
        v_pos, v_nrm, t_pos_idx, island_padding
    )  # (Nf*3,2), per-corner already

    individual_vertices = v_pos[t_pos_idx].reshape(-1, 3)
    n_corners = individual_vertices.shape[0]
    individual_faces = np.arange(n_corners, dtype=t_pos_idx.dtype).reshape(-1, 3)

    v_nrm_flat = compute_vertex_normal(individual_vertices, individual_faces)
    v_tng_flat = compute_vertex_tangent(
        individual_vertices, uv_flat, v_nrm_flat, individual_faces
    )

    return individual_vertices, individual_faces, uv_flat, v_nrm_flat, v_tng_flat
