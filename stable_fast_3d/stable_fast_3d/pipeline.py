"""
End-to-end single-image -> textured GLB mesh pipeline: everything in
`sf3d.system.SF3D.run_image` reimplemented in native MLX + numpy, no
PyTorch anywhere. Two stages:

1. Neural forward pass (camera embedder -> DINOv2 tokenizer -> transformer
   backbone -> post-processor) turns an image into `scene_codes`, a
   (3, 40, 384, 384) triplane.
2. Mesh finishing (marching-tetrahedra isosurface extraction, UV unwrap,
   texture baking, using the same triplane decoder queried at the baked
   texel positions) turns `scene_codes` into a textured GLB.

Simplification carried over from the original: roughness/metallic use fixed
constants rather than a real per-image CLIP estimate (SF3D's
`image_estimator` head is not ported - not needed for a raw textured mesh).
"""

import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import rembg
from PIL import Image

from .backbone import TwoStreamInterleaveTransformer
from .backbone import load_into as load_backbone
from .camera_embedder import LinearCameraEmbedder, TriplaneLearnablePositionalEmbedding
from .decoder import MaterialMLP
from .decoder import load_into as load_decoder
from .decoder import query_triplane
from .dinov2 import IMAGE_MEAN, IMAGE_STD, Dinov2Model
from .dinov2 import load_into as load_dinov2
from .isosurface import scene_codes_to_mesh
from .post_processor import OUT_CHANNELS, PixelShuffleUpsampleNetwork
from .post_processor import load_into as load_post_processor
from .texture_baker import get_mask, interpolate, rasterize
from .uv_unwrap import mesh_unwrap_uv
from .weights import get_weights_path

COND_IMAGE_SIZE = 512
BACKGROUND_COLOR = np.array([0.5, 0.5, 0.5], dtype=np.float32)
PLANE_SIZE = 96
TRIPLANE_CHANNELS = 1024
DEFAULT_DISTANCE = 1.6
DEFAULT_FOVY_DEG = 40.0
DEFAULT_ROUGHNESS = 1.0
DEFAULT_METALLIC = 0.0


def _default_cond_c2w(distance: float) -> np.ndarray:
    # sf3d.utils.default_cond_c2w, verbatim
    return np.array(
        [[0, 0, 1, distance], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )


def _intrinsic_normed(fov_deg: float, size: int) -> np.ndarray:
    # sf3d.utils.create_intrinsic_from_fov_deg + sf3d.models.utils.get_intrinsic_from_fov
    fov = np.deg2rad(fov_deg)
    focal = 0.5 * size / np.tan(0.5 * fov)
    intrinsic = np.eye(3, dtype=np.float32)
    intrinsic[0, 0] = focal
    intrinsic[1, 1] = focal
    intrinsic[0, 2] = size / 2.0
    intrinsic[1, 2] = size / 2.0
    normed = intrinsic.copy()
    normed[0, 0] /= size
    normed[1, 1] /= size
    normed[0, 2] /= size
    normed[1, 2] /= size
    return normed


def _detokenize(tokens: mx.array, plane_size: int, num_channels: int) -> mx.array:
    b, ct, nt = tokens.shape
    x = tokens.reshape(b, ct, 3, plane_size, plane_size)
    return x.transpose(0, 2, 1, 3, 4)  # B Np Ct Hp Wp


def _normalize(x: np.ndarray, axis=-1, eps=1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, eps, None)


def _dilate_fill(img: np.ndarray, mask: np.ndarray, iterations: int) -> np.ndarray:
    """numpy port of sf3d.models.utils.dilate_fill - grows the valid region
    by `iterations` texels, filling new texels with the mean color of their
    already-valid 3x3 neighbors. img: (H,W,3) float32. mask: (H,W) bool."""
    H, W = mask.shape
    old_mask = mask.astype(np.float32)
    old_img = img.astype(np.float32).copy()
    offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    def shift(arr, dy, dx):
        out = np.zeros_like(arr)
        ys = slice(max(0, dy), H + min(0, dy))
        xs = slice(max(0, dx), W + min(0, dx))
        ys_src = slice(max(0, -dy), H + min(0, -dy))
        xs_src = slice(max(0, -dx), W + min(0, -dx))
        out[ys, xs] = arr[ys_src, xs_src]
        return out

    for _ in range(iterations):
        new_mask = np.zeros_like(old_mask)
        for dy, dx in offsets:
            new_mask = np.maximum(new_mask, shift(old_mask, dy, dx))

        img_sum = np.zeros_like(old_img)
        mask_sum = np.zeros_like(old_mask)
        masked_img = old_img * old_mask[..., None]
        for dy, dx in offsets:
            img_sum += shift(masked_img, dy, dx)
            mask_sum += shift(old_mask, dy, dx)
        mean_color = img_sum / np.clip(mask_sum, 1, None)[..., None]

        diff_mask = (new_mask - old_mask)[..., None]
        old_img = old_img * (1 - diff_mask) + mean_color * diff_mask
        old_mask = new_mask

    return old_img


def _float32_to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


class StableFast3D:
    """Loads once (~4GB of weights), then call `.run(image, output_path)`
    per generation. Downloads and converts the checkpoint from Hugging Face
    on first use - see `weights.get_weights_path`."""

    def __init__(self, weights_path: Path = None):
        weights = mx.load(str(weights_path or get_weights_path()))

        self.camera_embedder = LinearCameraEmbedder()
        self.camera_embedder.linear.weight = weights["camera_embedder.linear.weight"]
        self.camera_embedder.linear.bias = weights["camera_embedder.linear.bias"]

        self.dino = Dinov2Model()
        load_dinov2(self.dino, weights)

        self.tokenizer = TriplaneLearnablePositionalEmbedding()
        self.tokenizer.embeddings = weights["tokenizer.embeddings"]

        self.backbone = TwoStreamInterleaveTransformer()
        load_backbone(self.backbone, weights)

        self.post_processor = PixelShuffleUpsampleNetwork()
        load_post_processor(self.post_processor, weights)

        self.decoder = MaterialMLP()
        load_decoder(self.decoder, weights)

        self._rembg_session = rembg.new_session(providers=["CPUExecutionProvider"])

    # ---- preprocessing (mirrors sf3d.utils.remove_background/resize_foreground
    # + sf3d.system.SF3D.prepare_image, all torch-free here) ----
    def _remove_background_and_crop(
        self, image: Image.Image, foreground_ratio: float
    ) -> Image.Image:
        if image.mode != "RGBA" or image.getextrema()[3][0] == 255:
            image = rembg.remove(image.convert("RGB"), session=self._rembg_session)
        mask_np = np.array(image)[:, :, -1]
        ys, xs = np.nonzero(mask_np > 127)
        if len(xs) == 0:
            return image.convert("RGBA")
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        h, w = y2 - y1, x2 - x1
        yc, xc = (y1 + y2) / 2, (x1 + x2) / 2
        scale = max(h, w) / foreground_ratio
        left, top = int(xc - scale / 2), int(yc - scale / 2)
        size = int(scale)
        canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        src = image.crop((left, top, left + size, top + size))
        canvas.paste(src, (0, 0))
        return canvas

    def prepare_image(
        self, image: Image.Image, foreground_ratio: float = 0.85, remove_bg: bool = True
    ) -> np.ndarray:
        """PIL image (any mode/size) -> (512,512,3) float32 rgb_cond in [0,1],
        matching SF3D.prepare_image after remove_background/resize_foreground."""
        if remove_bg:
            image = self._remove_background_and_crop(image, foreground_ratio)
        else:
            image = image.convert("RGBA")

        image = image.resize((COND_IMAGE_SIZE, COND_IMAGE_SIZE))
        arr = np.asarray(image).astype(np.float32) / 255.0
        arr = np.clip(arr, 0.0, 1.0)
        mask = arr[:, :, 3:4]
        rgb = BACKGROUND_COLOR[None, None, :] * (1 - mask) + arr[:, :, :3] * mask
        # ImageProcessor's F.interpolate(bilinear, antialias=True) at 512->512
        # is an identity resize (scale=1, no filtering applied) - skipped.
        return rgb.astype(np.float32)

    def image_to_scene_codes(
        self, image: Image.Image, foreground_ratio: float = 0.85, remove_bg: bool = True
    ) -> np.ndarray:
        rgb_cond = self.prepare_image(image, foreground_ratio, remove_bg)  # (512,512,3)

        c2w_cond = _default_cond_c2w(DEFAULT_DISTANCE)
        intrinsic_normed = _intrinsic_normed(DEFAULT_FOVY_DEG, COND_IMAGE_SIZE)
        c2w_mx = mx.array(c2w_cond).reshape(1, 1, 4, 4)
        intr_mx = mx.array(intrinsic_normed).reshape(1, 1, 3, 3)
        camera_embeds = self.camera_embedder(c2w_mx, intr_mx)  # (1,1,768)

        pixel_values = mx.array(rgb_cond)[None]  # (1,512,512,3)
        pixel_values = (pixel_values - IMAGE_MEAN) / IMAGE_STD
        modulation_cond = camera_embeds.reshape(1, 768)
        image_tokens = self.dino(pixel_values, modulation_cond)  # (1,1297,1024)

        triplane_tokens_init = self.tokenizer(1)  # (1,1024,27648)
        triplane_tokens_backbone = self.backbone(triplane_tokens_init, image_tokens)

        direct_codes = _detokenize(
            triplane_tokens_backbone, PLANE_SIZE, TRIPLANE_CHANNELS
        )  # (1,3,1024,96,96)
        b, np_, ci, hp, wp = direct_codes.shape
        x = direct_codes.reshape(b * np_, ci, hp, wp).transpose(0, 2, 3, 1)  # NHWC

        scene_codes_flat = self.post_processor(x)  # (B*3, 384, 384, 40) NHWC
        mx.eval(scene_codes_flat)

        scene_codes = np.array(scene_codes_flat).transpose(0, 3, 1, 2)  # NCHW
        scene_codes = scene_codes.reshape(
            b, np_, OUT_CHANNELS, scene_codes.shape[-2], scene_codes.shape[-1]
        )
        return scene_codes

    def finish_mesh(
        self,
        scene_codes_np: np.ndarray,
        output_glb_path: str,
        bake_resolution: int = 512,
    ) -> str:
        if scene_codes_np.ndim == 5:
            scene_codes_np = scene_codes_np[0]
        scene_codes_mx = mx.array(scene_codes_np)

        v_pos, faces = scene_codes_to_mesh(scene_codes_mx, self.decoder)
        if v_pos.shape[0] == 0:
            raise RuntimeError("empty mesh (isosurface threshold produced no geometry)")

        v_pos_f, t_idx_f, v_tex_f, v_nrm_f, v_tng_f = mesh_unwrap_uv(
            v_pos.astype(np.float64), faces
        )

        rast = rasterize(
            v_tex_f.astype(np.float32), t_idx_f.astype(np.int64), bake_resolution
        )
        bake_mask = get_mask(rast)

        pos_bake = interpolate(
            v_pos_f.astype(np.float32), rast, t_idx_f.astype(np.int64)
        )
        gb_pos = pos_bake[bake_mask]  # (N,3)

        query_pts = mx.array(gb_pos.astype(np.float32))
        tri_query = query_triplane(query_pts, scene_codes_mx)
        decoded = self.decoder(tri_query, ["features", "perturb_normal"])
        mx.eval(decoded["features"], decoded["perturb_normal"])
        albedo = np.array(decoded["features"])  # (N,3) in [0,1] (sigmoid)
        normal = _normalize(np.array(decoded["perturb_normal"]), axis=-1)

        nrm_bake = interpolate(
            v_nrm_f.astype(np.float32), rast, t_idx_f.astype(np.int64)
        )
        gb_nrm = _normalize(nrm_bake[bake_mask], axis=-1)

        f_albedo = np.zeros((bake_resolution, bake_resolution, 3), dtype=np.float32)
        f_albedo[bake_mask] = albedo

        tng_bake = interpolate(
            v_tng_f.astype(np.float32), rast, t_idx_f.astype(np.int64)
        )
        gb_tng = _normalize(tng_bake[bake_mask], axis=-1)
        gb_btng = _normalize(np.cross(gb_nrm, gb_tng), axis=-1)
        # tangent_matrix columns [tng, btng, nrm]; normal_tangent = tangent_matrix^T @ normal
        normal_tangent = (
            gb_tng * normal[:, 0:1] + gb_btng * normal[:, 1:2] + gb_nrm * normal[:, 2:3]
        )
        normal_tangent = np.clip(normal_tangent * 0.5 + 0.5, 0, 1)

        f_bump = np.zeros((bake_resolution, bake_resolution, 3), dtype=np.float32)
        f_bump[bake_mask] = normal_tangent

        iters = max(1, bake_resolution // 150)
        f_albedo = _dilate_fill(f_albedo, bake_mask, iters)
        f_bump = _dilate_fill(f_bump, bake_mask, iters)

        basecolor_tex = Image.fromarray(_float32_to_uint8(f_albedo)).convert("RGB")
        basecolor_tex.format = "JPEG"
        bump_tex = Image.fromarray(_float32_to_uint8(f_bump)).convert("RGB")
        bump_tex.format = "JPEG"

        import trimesh

        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=basecolor_tex,
            roughnessFactor=DEFAULT_ROUGHNESS,
            metallicFactor=DEFAULT_METALLIC,
            normalTexture=bump_tex,
        )
        tmesh = trimesh.Trimesh(
            vertices=v_pos_f,
            faces=t_idx_f,
            visual=trimesh.visual.texture.TextureVisuals(uv=v_tex_f, material=material),
        )
        rot = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
        tmesh.apply_transform(rot)
        tmesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0])
        )
        tmesh.invert()

        tmesh.export(output_glb_path, include_normals=True)
        return output_glb_path

    def run(
        self,
        image: Image.Image,
        output_glb_path: str,
        foreground_ratio: float = 0.85,
        remove_bg: bool = True,
        bake_resolution: int = 512,
        verbose: bool = False,
    ) -> str:
        """image (any size/mode) -> textured GLB at `output_glb_path`."""
        t0 = time.perf_counter()
        scene_codes = self.image_to_scene_codes(image, foreground_ratio, remove_bg)
        if verbose:
            print(f"[stable_fast_3d] neural pass: {time.perf_counter() - t0:.1f}s")

        t0 = time.perf_counter()
        out = self.finish_mesh(scene_codes, output_glb_path, bake_resolution)
        if verbose:
            print(f"[stable_fast_3d] mesh finishing: {time.perf_counter() - t0:.1f}s")
        return out
