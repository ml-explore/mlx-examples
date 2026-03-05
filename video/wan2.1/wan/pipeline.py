# Copyright © 2026 Apple Inc.

"""
Wan2.1 text-to-video pipeline.
"""

from typing import Optional, Tuple

import mlx.core as mx

from .sampler import FlowUniPCMultistepScheduler
from .utils import load_dit, load_t5, load_t5_tokenizer, load_vae


class WanT2VPipeline:
    def __init__(self, name: str = "t2v-1.3B", dtype: mx.Dtype = mx.bfloat16):
        self.dtype = dtype
        self.name = name
        self.vae_stride = (4, 8, 8)
        self.z_dim = 16
        self._null_context = None

        # Disable Metal buffer cache to prevent swap pressure
        mx.set_cache_limit(0)

        self.flow = load_dit(name)
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
    ):
        """
        Generator yielding latents at each denoising step.

        First yield: conditioning tuple (for mx.eval by caller)
        Subsequent yields: latent at each denoising step
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

        # Initial noise
        x_T = mx.random.normal(target_shape).astype(self.dtype)

        # Yield conditioning for controlled evaluation
        yield (x_T, context, context_null)

        # Denoising loop
        self.sampler.set_timesteps(num_steps, shift=shift)

        x_t = x_T
        for t in self.sampler.timesteps:
            t_val = t.reshape(1).astype(mx.float32)

            # Conditional forward
            noise_cond = self.flow(
                [x_t],
                t=t_val,
                context=[context],
            )[0]

            # CFG: skip unconditional pass when guidance <= 1.0
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
            x_t = self.sampler.step(noise_pred, t, x_t)
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
