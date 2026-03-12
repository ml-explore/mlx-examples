# Copyright © 2026 Apple Inc.

"""
Wan2.1 text-to-video and image-to-video pipeline.
"""

import logging
from typing import Optional, Tuple

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

from .sampler import FlowEulerDiscreteScheduler, FlowUniPCMultistepScheduler
from .utils import configs, load_clip, load_dit, load_t5, load_t5_tokenizer, load_vae

# Polynomial coefficients for TeaCache distance rescaling (calibrated per model).
# Each entry: (coefficients, ret_steps, cutoff_offset, use_projected_embedding)
_tea_coeffs = {  # from https://github.com/ModelTC/LightX2V/blob/main/configs/caching/teacache/wan_t2v_1_3b_tea_480p.json
    "t2v-1.3B": {
        "coeffs": [
            -5.21862437e04,
            9.23041404e03,
            -5.28275948e02,
            1.36987616e01,
            -4.99875664e-02,
        ],
        "ret_steps": 5,
        "use_e0": True,
    },
    "t2v-14B": {  # from https://github.com/ModelTC/LightX2V/blob/main/configs/caching/custom/wan_t2v_custom_14b.json
        "coeffs": [
            -5784.54975374,
            5449.50911966,
            -1811.16591783,
            256.27178429,
            -13.02252404,
        ],
        "ret_steps": 1,
        "use_e0": False,
    },
    "i2v-14B": {  # from https://github.com/ModelTC/LightX2V/blob/main/configs/caching/teacache/wan_i2v_tea_480p.json
        "coeffs": [
            2.57151496e05,
            -3.54229917e04,
            1.40286849e03,
            -1.35890334e01,
            1.32517977e-01,
        ],
        "ret_steps": 5,
        "use_e0": True,
    },
}


class WanPipeline:
    def __init__(
        self,
        name: str = "t2v-1.3B",
        dtype: mx.Dtype = mx.bfloat16,
        checkpoint: Optional[str] = None,
    ):
        self.dtype = dtype
        self.name = name
        self.vae_stride = (4, 8, 8)
        self.z_dim = 16
        self._null_context = None

        self.flow = load_dit(name, checkpoint=checkpoint)
        self.vae = load_vae(name)
        self.t5 = load_t5(name)
        self.t5_tokenizer = load_t5_tokenizer(name)
        self.clip = load_clip(name) if configs[name].repo_clip else None
        self.sampler = FlowUniPCMultistepScheduler()

    def ensure_models_are_loaded(self):
        params = [
            self.flow.parameters(),
            self.vae.parameters(),
            self.t5.parameters(),
        ]
        if self.clip is not None:
            params.append(self.clip.parameters())
        mx.eval(*params)

    def tokenize(self, text: str):
        return self.t5_tokenizer(text)

    def _encode_text(self, text: str) -> mx.array:
        """Encode text prompt with T5. Returns [512, 4096]."""
        tokens = self.tokenize(text)
        ids = tokens["input_ids"]
        mask = tokens["attention_mask"]
        embeddings = self.t5(ids, mask=mask)
        seq_len = int(mask.sum().item())
        context = embeddings[0, :seq_len, :]
        if seq_len < 512:
            padding = mx.zeros((512 - seq_len, context.shape[-1]))
            context = mx.concatenate([context, padding], axis=0)
        return context

    def _encode_null(self) -> mx.array:
        """Return cached empty-string T5 embedding for CFG."""
        if self._null_context is None:
            self._null_context = self._encode_text("")
        return self._null_context

    def _encode_clip(self, image_path: str) -> mx.array:
        """Encode image with CLIP. Returns [1, 257, 1280]."""
        from .clip import preprocess_clip_image

        img = preprocess_clip_image(image_path)
        return self.clip(img).astype(self.dtype)

    def _prepare_image_conditioning(
        self, image_path: str, size: Tuple[int, int], frame_num: int
    ) -> mx.array:
        """Prepare VAE-encoded first frame + temporal mask.

        Returns:
            y: [T', H', W', 20] conditioning tensor (channels-last)
        """
        from PIL import Image

        W, H = size
        T_latent = (frame_num - 1) // self.vae_stride[0] + 1
        H_latent = H // self.vae_stride[1]
        W_latent = W // self.vae_stride[2]

        # Load image, resize (short side) + center crop to target resolution
        img = Image.open(image_path).convert("RGB")
        iw, ih = img.size
        scale = max(W / iw, H / ih)
        rw, rh = round(iw * scale), round(ih * scale)
        img = img.resize((rw, rh), Image.BICUBIC)
        left = (rw - W) // 2
        top = (rh - H) // 2
        img = img.crop((left, top, left + W, top + H))

        # Normalize to [-1, 1]
        img_arr = np.array(img).astype(np.float32) / 255.0
        img_arr = (img_arr - 0.5) / 0.5
        img_tensor = mx.array(img_arr)  # [H, W, 3]

        # Build video: first frame = image, rest = zeros -> [F, H, W, 3]
        zeros = mx.zeros((frame_num - 1, H, W, 3))
        video = mx.concatenate([img_tensor[None], zeros], axis=0)

        # VAE encode -> [T', H', W', 16]
        vae_latent = self.vae.encode(video)

        # Build temporal mask -> [T', H', W', 4]
        msk_first = mx.ones((1, H_latent, W_latent, 4))
        msk_rest = mx.zeros((T_latent - 1, H_latent, W_latent, 4))
        msk = mx.concatenate([msk_first, msk_rest], axis=0)

        # Concat: [T', H', W', 4+16] = [T', H', W', 20]
        y = mx.concatenate([msk, vae_latent], axis=-1)
        return y.astype(self.dtype)

    def generate_latents(
        self,
        text: str,
        image_path: Optional[str] = None,
        negative_prompt: str = "",
        size: Tuple[int, int] = (832, 480),
        frame_num: int = 81,
        num_steps: int = 50,
        guidance: float = 5.0,
        shift: float = 5.0,
        seed: Optional[int] = None,
        teacache: float = 0.0,
        verbose: bool = False,
        denoising_step_list=None,
    ):
        """
        Generator yielding latents at each denoising step.

        First yield: conditioning tuple (for mx.eval by caller)
        Subsequent yields: latent at each denoising step

        Args:
            image_path: Path to input image (I2V mode). None for T2V.
            denoising_step_list: If provided, use Euler scheduler for
                step-distilled models (e.g. [1000, 750, 500, 250]).
        """
        if denoising_step_list is not None and teacache > 0:
            logger.warning(
                "TeaCache is not calibrated for distilled models; disabling."
            )
            teacache = 0.0

        if seed is not None:
            mx.random.seed(seed)

        W, H = size
        target_shape = (
            (frame_num - 1) // self.vae_stride[0] + 1,
            H // self.vae_stride[1],
            W // self.vae_stride[2],
            self.z_dim,
        )

        # Encode text
        context = self._encode_text(text)
        if negative_prompt:
            context_null = self._encode_text(negative_prompt)
        else:
            context_null = self._encode_null()

        # Image conditioning (I2V only)
        clip_features = None
        first_frame = None
        if image_path is not None and self.clip is not None:
            clip_features = self._encode_clip(image_path)
            first_frame = self._prepare_image_conditioning(image_path, size, frame_num)

        # Initial noise
        x_T = mx.random.normal(target_shape).astype(self.dtype)

        # Yield conditioning for controlled evaluation
        yield (x_T, context, context_null, clip_features, first_frame)

        # Denoising loop — choose sampler
        if denoising_step_list is not None:
            sampler = FlowEulerDiscreteScheduler()
            sampler.set_timesteps(denoising_step_list, shift=shift)
            num_steps = len(denoising_step_list)
        else:
            sampler = self.sampler
            sampler.set_timesteps(num_steps, shift=shift)

        # TeaCache state
        use_teacache = teacache > 0
        if use_teacache:
            tea_cfg = _tea_coeffs[self.name]
            coeffs = tea_cfg["coeffs"]
            ret_steps = tea_cfg["ret_steps"]
            use_e0 = tea_cfg["use_e0"]
            cutoff_steps = num_steps if use_e0 else num_steps - 1
            prev_e0 = None
            accum_cond = 0.0
            accum_uncond = 0.0
            prev_residual_cond = None
            prev_residual_uncond = None
            skipped_steps = 0

        flow = mx.compile(self.flow.__call__, inputs=[self.flow.state])

        x_t = x_T
        for step_idx, t in enumerate(sampler.timesteps):
            t_val = t.reshape(1).astype(mx.float32)

            if use_teacache:
                t_emb, e0 = self.flow.compute_time_embedding(t_val)
                mx.eval(t_emb, e0)

                must_compute = (
                    step_idx < ret_steps or step_idx >= cutoff_steps or prev_e0 is None
                )

                if not must_compute:
                    dist_emb = e0 if use_e0 else t_emb
                    raw_dist = (
                        mx.abs(dist_emb - prev_e0).mean()
                        / (mx.abs(prev_e0).mean() + 1e-8)
                    ).item()
                    rescaled = float(np.polyval(coeffs, raw_dist))
                    accum_cond += abs(rescaled)
                    accum_uncond += abs(rescaled)

                skip_cond = (
                    not must_compute
                    and accum_cond < teacache
                    and prev_residual_cond is not None
                )
                if skip_cond:
                    noise_cond, _ = flow(
                        x_t,
                        t=t_val,
                        context=context,
                        clip_fea=clip_features,
                        first_frame=first_frame,
                        block_residual=prev_residual_cond,
                        precomputed_time=(t_emb, e0),
                    )
                    skipped_steps += 1
                    if verbose:
                        logger.info(
                            f"Step {step_idx}/{num_steps}: skip "
                            f"(accum_cond={accum_cond:.4f})"
                        )
                else:
                    noise_cond, prev_residual_cond = flow(
                        x_t,
                        t=t_val,
                        context=context,
                        clip_fea=clip_features,
                        first_frame=first_frame,
                        precomputed_time=(t_emb, e0),
                    )
                    mx.eval(prev_residual_cond)
                    accum_cond = 0.0
                    if verbose:
                        logger.info(f"Step {step_idx}/{num_steps}: compute")

                if guidance > 1.0:
                    skip_uncond = (
                        not must_compute
                        and accum_uncond < teacache
                        and prev_residual_uncond is not None
                    )
                    if skip_uncond:
                        noise_uncond, _ = flow(
                            x_t,
                            t=t_val,
                            context=context_null,
                            clip_fea=clip_features,
                            first_frame=first_frame,
                            block_residual=prev_residual_uncond,
                            precomputed_time=(t_emb, e0),
                        )
                    else:
                        noise_uncond, prev_residual_uncond = flow(
                            x_t,
                            t=t_val,
                            context=context_null,
                            clip_fea=clip_features,
                            first_frame=first_frame,
                            precomputed_time=(t_emb, e0),
                        )
                        mx.eval(prev_residual_uncond)
                        accum_uncond = 0.0
                    noise_pred = noise_uncond + guidance * (noise_cond - noise_uncond)
                else:
                    noise_pred = noise_cond

                prev_e0 = e0 if use_e0 else t_emb

                if verbose and step_idx == num_steps - 1:
                    logger.info(
                        f"TeaCache: skipped {skipped_steps}/{num_steps} steps "
                        f"({100 * skipped_steps / num_steps:.0f}%)"
                    )
            else:
                # Standard path
                noise_cond, _ = flow(
                    x_t,
                    t=t_val,
                    context=context,
                    clip_fea=clip_features,
                    first_frame=first_frame,
                )

                if guidance > 1.0:
                    noise_uncond, _ = flow(
                        x_t,
                        t=t_val,
                        context=context_null,
                        clip_fea=clip_features,
                        first_frame=first_frame,
                    )
                    noise_pred = noise_uncond + guidance * (noise_cond - noise_uncond)
                else:
                    noise_pred = noise_cond

            # Scheduler step
            x_t = sampler.step(noise_pred, t, x_t)
            mx.async_eval(x_t)
            yield x_t

    def decode(self, latents: mx.array, compile_vae: bool = False) -> mx.array:
        """
        Decode latents to video frames.

        Args:
            latents: [F, H, W, C] latent tensor (channels-last)
            compile_vae: If True, compile the VAE decoder for frames 1+

        Returns:
            [F, H, W, C] video tensor in [-1, 1] (channels-last)
        """
        return self.vae.decode(latents, compile=compile_vae)
