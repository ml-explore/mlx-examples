"""
Marching tetrahedra isosurface extraction, ported from SF3D's
`sf3d/models/isosurface.py` (`MarchingTetrahedraHelper`). Reuses this
package's `query_triplane` + `MaterialMLP` decoder to get density/
vertex_offset at the ~536k grid points, then extracts the mesh.

MLX has no `unique(..., return_inverse=True)` (needed for the shared-edge-
vertex dedup step), so that step runs in plain numpy instead. It's a
one-shot ~15-20k-triangle CPU op either way, not the inference bottleneck.
"""

import mlx.core as mx
import numpy as np

from .decoder import RADIUS, MaterialMLP, query_triplane
from .weights import get_tets_path

ISOSURFACE_RESOLUTION = 160
ISOSURFACE_THRESHOLD = 10.0
POINTS_RANGE = (0.0, 1.0)
BBOX = (-RADIUS, RADIUS)

# sf3d/models/isosurface.py's MarchingTetrahedraHelper constant tables, verbatim.
TRIANGLE_TABLE = np.array(
    [
        [-1, -1, -1, -1, -1, -1],
        [1, 0, 2, -1, -1, -1],
        [4, 0, 3, -1, -1, -1],
        [1, 4, 2, 1, 3, 4],
        [3, 1, 5, -1, -1, -1],
        [2, 3, 0, 2, 5, 3],
        [1, 4, 0, 1, 5, 4],
        [4, 2, 5, -1, -1, -1],
        [4, 5, 2, -1, -1, -1],
        [4, 1, 0, 4, 5, 1],
        [3, 2, 0, 3, 5, 2],
        [1, 3, 5, -1, -1, -1],
        [4, 1, 2, 4, 3, 1],
        [3, 0, 4, -1, -1, -1],
        [2, 0, 1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1],
    ],
    dtype=np.int64,
)
NUM_TRIANGLES_TABLE = np.array(
    [0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=np.int64
)
BASE_TET_EDGES = np.array([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=np.int64)


def scale_np(x: np.ndarray, inp_scale, tgt_scale) -> np.ndarray:
    x = (x - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    return x * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]


def load_tets(path=None):
    d = np.load(path or get_tets_path())
    return d["vertices"].astype(np.float32), d["indices"].astype(np.int64)


def _sort_edges(edges: np.ndarray) -> np.ndarray:
    order = (edges[:, 0] > edges[:, 1]).astype(np.int64)[:, None]
    a = np.take_along_axis(edges, order, axis=1)
    b = np.take_along_axis(edges, 1 - order, axis=1)
    return np.concatenate([a, b], axis=1)


def marching_tetrahedra(
    pos_nx3: np.ndarray, sdf_n: np.ndarray, tet_fx4: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Verbatim port of MarchingTetrahedraHelper._forward. pos_nx3: (Nv,3)
    grid vertex positions (already deformation-adjusted, local [0,1] space).
    sdf_n: (Nv,) signed distance (density - threshold). tet_fx4: (Ntet,4)
    int tetrahedra indices. Returns (verts (Nv2,3), faces (Nf,3))."""
    occ_n = sdf_n > 0
    occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
    occ_sum = occ_fx4.sum(-1)
    valid_tets = (occ_sum > 0) & (occ_sum < 4)
    occ_sum = occ_sum[valid_tets]

    all_edges = tet_fx4[valid_tets][:, BASE_TET_EDGES].reshape(-1, 2)
    all_edges = _sort_edges(all_edges)
    unique_edges, idx_map = np.unique(all_edges, axis=0, return_inverse=True)
    idx_map = idx_map.reshape(-1)

    mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
    mapping = -np.ones(unique_edges.shape[0], dtype=np.int64)
    mapping[mask_edges] = np.arange(mask_edges.sum(), dtype=np.int64)
    idx_map = mapping[idx_map]

    interp_v = unique_edges[mask_edges]
    edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1, 2, 3)
    edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1, 2, 1).copy()
    edges_to_interp_sdf[:, -1] *= -1

    denominator = edges_to_interp_sdf.sum(1, keepdims=True)
    edges_to_interp_sdf = np.flip(edges_to_interp_sdf, axis=1) / denominator
    verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

    idx_map = idx_map.reshape(-1, 6)

    v_id = np.power(2, np.arange(4, dtype=np.int64))
    tetindex = (occ_fx4[valid_tets] * v_id[None, :]).sum(-1)
    num_triangles = NUM_TRIANGLES_TABLE[tetindex]

    sel1 = num_triangles == 1
    sel2 = num_triangles == 2
    faces_1 = np.take_along_axis(
        idx_map[sel1], TRIANGLE_TABLE[tetindex[sel1]][:, :3], axis=1
    ).reshape(-1, 3)
    faces_2 = np.take_along_axis(
        idx_map[sel2], TRIANGLE_TABLE[tetindex[sel2]][:, :6], axis=1
    ).reshape(-1, 3)
    faces = np.concatenate([faces_1, faces_2], axis=0)

    return verts.astype(np.float32), faces.astype(np.int64)


def scene_codes_to_mesh(
    scene_codes: mx.array, material_mlp: MaterialMLP, batch_points: int = 131072
) -> tuple[np.ndarray, np.ndarray]:
    """scene_codes: (3, Cp, Hp, Wp) MLX triplane for ONE mesh (matches
    query_triplane's expected single-batch-element input). Returns
    (verts (Nv,3) world-space float32, faces (Nf,3) int64) - same convention
    as sf3d.system.SF3D.triplane_to_meshes()[i], pre-remesh."""
    raw_grid_vertices, tet_indices = load_tets()
    grid_vertices_world = scale_np(raw_grid_vertices, POINTS_RANGE, BBOX)

    n = grid_vertices_world.shape[0]
    densities = []
    offsets = []
    for start in range(0, n, batch_points):
        chunk = mx.array(grid_vertices_world[start : start + batch_points])
        feats = query_triplane(chunk, scene_codes)
        decoded = material_mlp(feats, ["density", "vertex_offset"])
        mx.eval(decoded["density"], decoded["vertex_offset"])
        densities.append(np.array(decoded["density"]))
        offsets.append(np.array(decoded["vertex_offset"]))
    density = np.concatenate(densities, axis=0).reshape(-1)
    vertex_offset = np.concatenate(offsets, axis=0).reshape(-1, 3)

    sdf = density - ISOSURFACE_THRESHOLD

    deform_scale = (POINTS_RANGE[1] - POINTS_RANGE[0]) / ISOSURFACE_RESOLUTION
    grid_vertices_local = raw_grid_vertices + deform_scale * np.tanh(vertex_offset)

    v_pos, faces = marching_tetrahedra(grid_vertices_local, sdf, tet_indices)
    v_pos_world = scale_np(v_pos, POINTS_RANGE, BBOX)

    return v_pos_world, faces
