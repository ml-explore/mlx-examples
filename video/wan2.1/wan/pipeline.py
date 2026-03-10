# Copyright © 2026 Apple Inc.

"""
Wan2.1 text-to-video and image-to-video pipelines.
"""

import logging
from typing import Optional, Tuple

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

from .sampler import FlowEulerDiscreteScheduler, FlowUniPCMultistepScheduler
from .utils import load_clip, load_dit, load_t5, load_t5_tokenizer, load_vae

# Polynomial coefficients for TeaCache distance rescaling (calibrated per model).
# Each entry is (use_ret_steps=True coefficients, use_ret_steps=False coefficients).
_tea_coeffs = {
    "t2v-1.3B": (
        [-5.21862437e04, 9.23041404e03, -5.28275948e02, 1.36987616e01, -4.99875664e-02],
        [2.39676752e03, -1.31110545e03, 2.01331979e02, -8.29855975e00, 1.37887774e-01],
    ),
    "t2v-14B": (
        [-3.03318725e05, 4.90537029e04, -2.65530556e03, 5.87365115e01, -3.15583525e-01],
        [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404],
    ),
}


class WanT2VPipeline:
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

        # Disable Metal buffer cache to prevent swap pressure
        mx.set_cache_limit(0)

        self.flow = load_dit(name, checkpoint=checkpoint)
        self.vae = load_vae(name)
        self.t5 = load_t5(name)
        self.t5_tokenizer = load_t5_tokenizer(name)
        self.sampler = FlowUniPCMultistepScheduler()

    def ensure_models_are_loaded(self):
        mx.eval(
            self.flow.parameters(),
            self.vae.parameters(),
            self.t5.parameters(),
        )

    def tokenize(self, text: str):
        return self.t5_tokenizer(text)

    def _encode_text(self, text: str) -> mx.array:
        """Encode text prompt with T5. Returns [L, 4096] (variable length)."""
        tokens = self.tokenize(text)
        ids = tokens["input_ids"]
        mask = tokens["attention_mask"]
        embeddings = self.t5(ids, mask=mask)
        # Truncate to actual tokens then re-pad to 512
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

    def generate_latents(
        self,
        text: str,
        negative_prompt: str = "",
        size: Tuple[int, int] = (832, 480),
        frame_num: int = 81,
        num_steps: int = 50,
        guidance: float = 5.0,
        shift: float = 5.0,
        seed: Optional[int] = None,
        teacache: float = 0.0,
        use_ret_steps: bool = True,
        verbose: bool = False,
        denoising_step_list=None,
    ):
        """
        Generator yielding latents at each denoising step.

        First yield: conditioning tuple (for mx.eval by caller)
        Subsequent yields: latent at each denoising step

        Args:
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
            self.z_dim,
            (frame_num - 1) // self.vae_stride[0] + 1,
            H // self.vae_stride[1],
            W // self.vae_stride[2],
        )

        # Encode text
        context = self._encode_text(text)
        if negative_prompt:
            context_null = self._encode_text(negative_prompt)
        else:
            context_null = self._encode_null()

        # Initial noise
        x_T = mx.random.normal(target_shape).astype(self.dtype)

        # Yield conditioning for controlled evaluation
        yield (x_T, context, context_null)

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
            coeffs = _tea_coeffs[self.name][0 if use_ret_steps else 1]
            ret_steps = 5 if use_ret_steps else 1
            cutoff_steps = num_steps if use_ret_steps else num_steps - 1
            prev_e0 = None
            accum_cond = 0.0
            accum_uncond = 0.0
            prev_residual_cond = None
            prev_residual_uncond = None
            skipped_steps = 0

        x_t = x_T
        for step_idx, t in enumerate(sampler.timesteps):
            t_val = t.reshape(1).astype(mx.float32)

            if use_teacache:
                # Precompute time embedding once per step
                t_emb, e0 = self.flow.compute_time_embedding(t_val)
                mx.eval(t_emb, e0)

                # Determine whether to run full forward pass
                # Always compute first and last steps
                must_compute = (
                    step_idx < ret_steps or step_idx >= cutoff_steps or prev_e0 is None
                )

                if not must_compute:
                    # Relative L1 distance with polynomial rescaling
                    dist_emb = e0 if use_ret_steps else t_emb
                    raw_dist = (
                        mx.abs(dist_emb - prev_e0).mean()
                        / (mx.abs(prev_e0).mean() + 1e-8)
                    ).item()
                    rescaled = float(np.polyval(coeffs, raw_dist))
                    accum_cond += abs(rescaled)
                    accum_uncond += abs(rescaled)

                # Conditional forward
                skip_cond = (
                    use_teacache
                    and not must_compute
                    and accum_cond < teacache
                    and prev_residual_cond is not None
                )
                if skip_cond:
                    noise_cond = self.flow(
                        [x_t],
                        t=t_val,
                        context=[context],
                        block_residual=prev_residual_cond,
                        precomputed_time=(t_emb, e0),
                    )[0]
                    skipped_steps += 1
                    if verbose:
                        logger.info(
                            f"Step {step_idx}/{num_steps}: skip "
                            f"(accum_cond={accum_cond:.4f})"
                        )
                else:
                    noise_cond = self.flow(
                        [x_t],
                        t=t_val,
                        context=[context],
                        precomputed_time=(t_emb, e0),
                    )[0]
                    # Set by model.__call__ — TeaCache block-residual caching
                    prev_residual_cond = self.flow._last_block_residual
                    mx.eval(prev_residual_cond)  # Materialize to release graph
                    accum_cond = 0.0
                    if verbose:
                        logger.info(f"Step {step_idx}/{num_steps}: compute")

                # Unconditional forward (CFG)
                if guidance > 1.0:
                    skip_uncond = (
                        not must_compute
                        and accum_uncond < teacache
                        and prev_residual_uncond is not None
                    )
                    if skip_uncond:
                        noise_uncond = self.flow(
                            [x_t],
                            t=t_val,
                            context=[context_null],
                            block_residual=prev_residual_uncond,
                            precomputed_time=(t_emb, e0),
                        )[0]
                    else:
                        noise_uncond = self.flow(
                            [x_t],
                            t=t_val,
                            context=[context_null],
                            precomputed_time=(t_emb, e0),
                        )[0]
                        prev_residual_uncond = self.flow._last_block_residual
                        mx.eval(prev_residual_uncond)
                        accum_uncond = 0.0
                    noise_pred = noise_uncond + guidance * (noise_cond - noise_uncond)
                else:
                    noise_pred = noise_cond

                prev_e0 = e0 if use_ret_steps else t_emb

                if verbose and step_idx == num_steps - 1:
                    logger.info(
                        f"TeaCache: skipped {skipped_steps}/{num_steps} steps "
                        f"({100 * skipped_steps / num_steps:.0f}%)"
                    )
            else:
                # Standard path (no TeaCache)
                noise_cond = self.flow(
                    [x_t],
                    t=t_val,
                    context=[context],
                )[0]

                if guidance > 1.0:
                    noise_uncond = self.flow(
                        [x_t],
                        t=t_val,
                        context=[context_null],
                    )[0]
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
            latents: [C, F, H, W] latent tensor
            compile_vae: If True, compile the VAE decoder for frames 1+

        Returns:
            [C, F, H, W] video tensor in [-1, 1]
        """
        return self.vae.decode(latents, compile=compile_vae)


class WanI2VPipeline:
    def __init__(
        self,
        name: str = "i2v-14B",
        dtype: mx.Dtype = mx.bfloat16,
        checkpoint: Optional[str] = None,
    ):
        self.dtype = dtype
        self.name = name
        self.vae_stride = (4, 8, 8)
        self.z_dim = 16
        self._null_context = None

        mx.set_cache_limit(0)

        self.flow = load_dit(name, checkpoint=checkpoint)
        self.vae = load_vae(name)
        self.t5 = load_t5(name)
        self.t5_tokenizer = load_t5_tokenizer(name)
        self.clip = load_clip(name)
        self.sampler = FlowUniPCMultistepScheduler()

    def ensure_models_are_loaded(self):
        mx.eval(
            self.flow.parameters(),
            self.vae.parameters(),
            self.t5.parameters(),
            self.clip.parameters(),
        )

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
            y: [20, T', H', W'] conditioning tensor (4-ch mask + 16-ch latent)
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

        # Build video: first frame = image, rest = zeros -> [3, F, H, W]
        img_chw = img_tensor.transpose(2, 0, 1)  # [3, H, W]
        zeros = mx.zeros((3, frame_num - 1, H, W))
        video = mx.concatenate([img_chw[:, None, :, :], zeros], axis=1)  # [3, F, H, W]

        # VAE encode
        vae_latent = self.vae.encode(video)  # [16, T', H', W']

        # Build temporal mask:
        # 1. mask [1, F, H', W'] — first frame=1, rest=0
        # 2. Repeat first position 4x, concat with rest
        # 3. Reshape to [4, T', H', W']
        msk = mx.concatenate(
            [
                mx.ones((1, 1, H_latent, W_latent)),
                mx.zeros((1, frame_num - 1, H_latent, W_latent)),
            ],
            axis=1,
        )
        first_repeated = mx.repeat(msk[:, 0:1], repeats=4, axis=1)
        msk = mx.concatenate([first_repeated, msk[:, 1:]], axis=1)
        msk = msk.reshape(1, T_latent, 4, H_latent, W_latent)
        msk = msk.transpose(0, 2, 1, 3, 4)[0]  # [4, T', H', W']

        # Concat: [4 + 16, T', H', W'] = [20, T', H', W']
        y = mx.concatenate([msk, vae_latent], axis=0)
        return y.astype(self.dtype)

    def generate_latents(
        self,
        text: str,
        image_path: str,
        negative_prompt: str = "",
        size: Tuple[int, int] = (832, 480),
        frame_num: int = 81,
        num_steps: int = 40,
        guidance: float = 5.0,
        shift: float = 3.0,
        seed: Optional[int] = None,
        verbose: bool = False,
        denoising_step_list=None,
    ):
        """
        Generator yielding latents at each denoising step.

        First yield: conditioning tuple (for mx.eval by caller)
        Subsequent yields: latent at each denoising step

        Args:
            denoising_step_list: If provided, use Euler scheduler for
                step-distilled models (e.g. [1000, 750, 500, 250]).
        """
        if seed is not None:
            mx.random.seed(seed)

        W, H = size
        target_shape = (
            self.z_dim,
            (frame_num - 1) // self.vae_stride[0] + 1,
            H // self.vae_stride[1],
            W // self.vae_stride[2],
        )

        # Encode text
        context = self._encode_text(text)
        if negative_prompt:
            context_null = self._encode_text(negative_prompt)
        else:
            context_null = self._encode_null()

        # Encode image with CLIP
        clip_features = self._encode_clip(image_path)

        # Prepare VAE image conditioning
        y = self._prepare_image_conditioning(image_path, size, frame_num)

        # Initial noise
        x_T = mx.random.normal(target_shape).astype(self.dtype)

        # Yield conditioning for controlled evaluation
        yield (x_T, context, context_null, clip_features, y)

        # Denoising loop — choose sampler
        if denoising_step_list is not None:
            sampler = FlowEulerDiscreteScheduler()
            sampler.set_timesteps(denoising_step_list, shift=shift)
            num_steps = len(denoising_step_list)
        else:
            sampler = self.sampler
            sampler.set_timesteps(num_steps, shift=shift)

        x_t = x_T
        for step_idx, t in enumerate(sampler.timesteps):
            t_val = t.reshape(1).astype(mx.float32)

            # Conditional forward (clip_fea and y used in both passes)
            noise_cond = self.flow(
                [x_t],
                t=t_val,
                context=[context],
                clip_fea=clip_features,
                y=[y],
            )[0]

            if guidance > 1.0:
                noise_uncond = self.flow(
                    [x_t],
                    t=t_val,
                    context=[context_null],
                    clip_fea=clip_features,
                    y=[y],
                )[0]
                noise_pred = noise_uncond + guidance * (noise_cond - noise_uncond)
            else:
                noise_pred = noise_cond

            # Scheduler step
            x_t = sampler.step(noise_pred, t, x_t)
            mx.async_eval(x_t)
            yield x_t

    def decode(self, latents: mx.array, compile_vae: bool = False) -> mx.array:
        """Decode latents to video frames."""
        return self.vae.decode(latents, compile=compile_vae)
