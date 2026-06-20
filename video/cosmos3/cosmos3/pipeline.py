"""Generation pipeline for Cosmos 3 Nano on MLX.

Wires the MoT transformer (generation path), scheduler, timestep
embedding, video VAE decoder, and audio decoder into a complete
text-to-image/video and image-to-video generation flow.
"""

import json
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .model import Cosmos3Config, Cosmos3Model
from .scheduler import UniPCScheduler
from .timestep import TimestepEmbedding, apply_timestep_to_noisy_tokens


# Cosmos3 Nano VAE latent config (Wan2.2 AutoencoderKL)
Z_DIM = 48
PATCH_SIZE = 2
LATENTS_MEAN = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.157, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.123,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.052, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
]
LATENTS_STD = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.499, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.06, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
]

_SYSTEM_PROMPT_IMAGE = "You are a helpful assistant who will generate images from a given prompt."
_SYSTEM_PROMPT_VIDEO = "You are a helpful assistant who will generate videos from a given prompt."


class Cosmos3GenerationPipeline:
    """End-to-end generation pipeline: text → image/video (+ audio).

    Orchestrates:
    1. Text tokenization and embedding
    2. Noise latent preparation
    3. Denoising loop (MoT generation path + scheduler)
    4. VAE decode to pixels
    5. Optional audio decode
    """

    def __init__(
        self,
        model: Cosmos3Model,
        tokenizer,
        model_dir: Optional[str | Path] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.scheduler = UniPCScheduler()
        self._model_dir = Path(model_dir) if model_dir is not None else None

        # Load reference negative prompt if available
        self._negative_prompt_text = None
        if model_dir is not None:
            neg_path = Path(model_dir) / "assets" / "negative_prompt.json"
            if neg_path.exists():
                with open(neg_path) as f:
                    self._negative_prompt_text = json.dumps(json.load(f))

        # Pre-compute latent normalization tensors
        self._latents_mean = mx.array(LATENTS_MEAN)
        self._latents_std = mx.array(LATENTS_STD)

    def _build_position_ids(
        self,
        text_len: int,
        grid_t: int,
        grid_h: int,
        grid_w: int,
        audio_tokens: Optional[mx.array] = None,
    ) -> mx.array:
        """Build 3D mRoPE position IDs for text + generation tokens.

        Matches the position ID construction inside diffusion_forward.
        """
        # Text positions
        text_pos = mx.arange(text_len, dtype=mx.float32)[None, :]
        text_position_ids = mx.stack([text_pos, text_pos, text_pos])

        # Generation positions (FPS-modulated)
        # Large temporal margin separates text and vision in mRoPE space.
        # HF pipeline chains text_mrope → vision_mrope with offset = text_len,
        # but the model may have been trained with a larger separation.
        temporal_margin = 15000
        temporal_offset = float(text_len + temporal_margin)
        fps = 24.0
        video_tcf = 4
        base_fps = 24.0
        tps = fps / video_tcf
        base_tps = base_fps / video_tcf

        frame_indices = mx.arange(grid_t, dtype=mx.float32)
        scaled_t = frame_indices / tps * base_tps + temporal_offset
        t_idx = mx.broadcast_to(scaled_t.reshape(-1, 1), (grid_t, grid_h * grid_w)).reshape(1, -1)

        h_idx = mx.arange(grid_h, dtype=mx.float32).reshape(1, -1, 1)
        h_idx = mx.broadcast_to(h_idx, (grid_t, grid_h, grid_w)).reshape(1, -1)

        w_idx = mx.arange(grid_w, dtype=mx.float32).reshape(1, 1, -1)
        w_idx = mx.broadcast_to(w_idx, (grid_t, grid_h, grid_w)).reshape(1, -1)

        gen_position_ids = mx.stack([t_idx, h_idx, w_idx])

        # Audio positions
        if audio_tokens is not None:
            sound_len = audio_tokens.shape[1]
            audio_tps = fps / 1.0
            audio_base_tps = base_fps / 1.0
            audio_frame_indices = mx.arange(sound_len, dtype=mx.float32)
            audio_scaled_t = audio_frame_indices / audio_tps * audio_base_tps + temporal_offset
            audio_t_idx = audio_scaled_t.reshape(1, -1)
            audio_h_idx = mx.zeros((1, sound_len), dtype=mx.float32)
            audio_w_idx = mx.zeros((1, sound_len), dtype=mx.float32)
            audio_position_ids = mx.stack([audio_t_idx, audio_h_idx, audio_w_idx])
            gen_position_ids = mx.concatenate([gen_position_ids, audio_position_ids], axis=2)

        return mx.concatenate([text_position_ids, gen_position_ids], axis=2)

    def _prepare_noise_latents(
        self,
        num_frames: int = 1,
        height: int = 512,
        width: int = 512,
        z_dim: int = 48,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> mx.array:
        """Prepare initial noise latents.

        Returns:
            [1, T_lat, H_lat, W_lat, z_dim] noise tensor (channels-last)
        """
        # Compute latent dimensions
        t_lat = max(1, num_frames // 4)  # 4x temporal compression
        h_lat = height // 16  # 16x spatial compression
        w_lat = width // 16

        noise = mx.random.normal((1, t_lat, h_lat, w_lat, z_dim)).astype(dtype)
        return noise

    def _patchify_latents(self, latents: mx.array) -> mx.array:
        """Convert VAE latents to patch tokens for the transformer.

        Input: [batch, T, H, W, z_dim]
        Output: [batch, num_patches, patch_latent_dim]

        With patch_size=2: each 2x2 spatial region becomes one token.
        patch_latent_dim = z_dim * patch_size * patch_size = 48 * 4 = 192

        If H or W are not divisible by patch_size, zero-pads to the next
        multiple (matching HF's _patchify_and_pack_latents). The padding
        only exists during the transformer forward; _unpatchify_latents
        crops back to original dims.
        """
        batch, t, h, w, z = latents.shape
        p = PATCH_SIZE

        h_p = (h + p - 1) // p  # ceil division
        w_p = (w + p - 1) // p
        h_padded = h_p * p
        w_padded = w_p * p

        # Zero-pad if needed (use concatenation, not .at[].add())
        if h_padded != h:
            pad_h = mx.zeros((batch, t, h_padded - h, w, z), dtype=latents.dtype)
            latents = mx.concatenate([latents, pad_h], axis=2)
        if w_padded != w:
            pad_w = mx.zeros((batch, t, latents.shape[2], w_padded - w, z), dtype=latents.dtype)
            latents = mx.concatenate([latents, pad_w], axis=3)

        # [B, T, H_pad//p, p, W_pad//p, p, z] -> [B, T*H_p*W_p, p*p*z]
        x = latents.reshape(batch, t, h_p, p, w_p, p, z)
        x = mx.transpose(x, (0, 1, 2, 4, 3, 5, 6))  # [B, T, H_p, W_p, p, p, z]
        x = x.reshape(batch, t * h_p * w_p, p * p * z)

        return x

    def _unpatchify_latents(
        self, tokens: mx.array, t: int, h_p: int, w_p: int,
        h_orig: int = 0, w_orig: int = 0,
    ) -> mx.array:
        """Convert patch tokens back to VAE latent shape.

        Input: [batch, num_patches, patch_latent_dim]
        Output: [batch, T, H_orig, W_orig, z_dim]

        Crops padded dimensions back to h_orig × w_orig when provided.
        """
        batch = tokens.shape[0]
        p = PATCH_SIZE
        z = Z_DIM

        x = tokens.reshape(batch, t, h_p, w_p, p, p, z)
        x = mx.transpose(x, (0, 1, 2, 4, 3, 5, 6))  # [B, T, H_p, p, W_p, p, z]
        x = x.reshape(batch, t, h_p * p, w_p * p, z)

        # Crop back to original dims (remove padding)
        if h_orig > 0 and w_orig > 0:
            x = x[:, :, :h_orig, :w_orig, :]

        return x

    def _encode_conditioning_image(
        self,
        image: np.ndarray,
        num_frames: int,
        height: int,
        width: int,
    ) -> mx.array:
        """Encode a conditioning image to normalized VAE latents.

        Matches HF Cosmos3OmniPipeline: builds a full video tensor with the
        conditioning frame repeated at every temporal position, then encodes
        the entire tensor through the VAE. The temporal causal convolutions
        produce per-frame latents that depend on preceding frames — frame 0
        sees zero-padded context while frames 1+ see frame 0's features.
        This is critical for i2v quality: tiling a single-frame encode
        produces identical latents at every position, but the model expects
        temporally-varying latents from the causal conv processing.

        Args:
            image: [H, W, 3] uint8 or float32 in [0, 1]
            num_frames: total video frames
            height: target height
            width: target width

        Returns:
            [1, T_lat, H_lat, W_lat, z_dim] normalized latents (channels-last)
        """
        from .encode_vae import encode_video

        vae_dir = str(self._model_dir / "vae")

        # Resize image to target resolution
        from PIL import Image as PILImage
        if isinstance(image, np.ndarray):
            pil_img = PILImage.fromarray(
                (image * 255).astype(np.uint8) if image.dtype == np.float32 else image
            )
        else:
            pil_img = image
        pil_img = pil_img.resize((width, height), PILImage.LANCZOS)
        image_np = np.array(pil_img).astype(np.float32) / 255.0

        if num_frames == 1:
            # Single image: encode directly
            return encode_video(image_np, vae_dir)

        # Build full video tensor: conditioning frame repeated at all positions.
        # HF: vision_tensor[:,:,0] = frame; vision_tensor[:,:,1:] = frame.repeat()
        # [T, H, W, 3] in [0, 1]
        video_tensor = np.stack([image_np] * num_frames, axis=0)
        return encode_video(video_tensor, vae_dir)

    def generate(
        self,
        prompt: str,
        num_frames: int = 1,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 6.0,
        seed: Optional[int] = None,
        enable_audio: bool = False,
        image: Optional[Union[np.ndarray, "PILImage"]] = None,
        negative_prompt: Optional[str] = None,
    ) -> dict:
        """Generate image/video from text prompt, optionally conditioned on an image.

        Args:
            prompt: text description
            num_frames: number of video frames (1 = single image)
            height: output height in pixels
            width: output width in pixels
            num_inference_steps: denoising steps
            guidance_scale: classifier-free guidance strength
            seed: random seed for reproducibility
            enable_audio: whether to generate audio alongside video
            image: optional conditioning image for i2v. When provided, frame 0
                is anchored to this image and the remaining frames are denoised
                freely. Can be numpy array [H,W,3] (uint8 or float32) or PIL Image.
            negative_prompt: optional negative prompt text. If None, uses the
                model's built-in negative prompt from assets/negative_prompt.json.

        Returns:
            dict with 'latents' (normalized, use decode_latents() to decode),
            'video' (if VAE available), 'audio_latents' (if enable_audio),
            'audio' (if audio decoded)
        """
        if seed is not None:
            mx.random.seed(seed)

        dtype = mx.bfloat16

        has_image_condition = image is not None and num_frames > 1

        # 1. Tokenize prompt
        is_image = (num_frames == 1)
        system_msg = _SYSTEM_PROMPT_IMAGE if is_image else _SYSTEM_PROMPT_VIDEO

        # Build user content with resolution/duration suffix
        if is_image:
            user_content = prompt + f" This image is of {height}x{width} resolution."
        else:
            fps = 24
            duration = num_frames / fps
            user_content = (prompt
                + f" The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
                + f" This video is of {height}x{width} resolution.")

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = self.tokenizer.encode(text)
        # Append EOS + vision_start sentinel (tells model generation follows)
        eos_id = self.tokenizer.eos_token_id
        vision_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        tokens = tokens + [eos_id, vision_start_id]
        cond_ids = mx.array([tokens])

        # Unconditional prompt for CFG
        neg_prompt = negative_prompt if negative_prompt is not None else (self._negative_prompt_text or "")
        if is_image:
            neg_suffix = f" This image is not of {height}x{width} resolution."
        else:
            neg_suffix = (f" The video is not {duration:.1f} seconds long and is not of {fps:.0f} FPS."
                + f" This video is not of {height}x{width} resolution.")
        neg_content = neg_prompt + neg_suffix

        uncond_messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": neg_content},
        ]
        uncond_text = self.tokenizer.apply_chat_template(
            uncond_messages, tokenize=False, add_generation_prompt=True,
        )
        uncond_tokens = self.tokenizer.encode(uncond_text)
        uncond_tokens = uncond_tokens + [eos_id, vision_start_id]
        uncond_ids = mx.array([uncond_tokens])

        # 2. Prepare noise latents
        z_dim = Z_DIM
        latents = self._prepare_noise_latents(
            num_frames, height, width, z_dim, dtype
        )
        t_lat = latents.shape[1]
        h_lat = latents.shape[2]
        w_lat = latents.shape[3]

        # Patchify grid: ceil division to match HF's zero-padding patchify.
        # Latents stay at original h_lat × w_lat between denoising steps.
        # Padding to patch-aligned dims only happens inside _patchify_latents,
        # and _unpatchify_latents crops back to h_lat × w_lat.
        p = PATCH_SIZE
        h_p = (h_lat + p - 1) // p  # ceil: 45→23 at 720p
        w_p = (w_lat + p - 1) // p

        # 2a. Image-to-video conditioning: encode image, create mask, mix
        # Vision condition mask: [T_lat, 1, 1] — 1.0 for conditioned frames, 0.0 for noisy
        vision_condition_mask = mx.zeros((t_lat, 1, 1))
        if has_image_condition:
            print("  Encoding conditioning image...")
            cond_latents = self._encode_conditioning_image(
                image, num_frames, height, width
            ).astype(dtype)
            # Crop to match noise latent temporal/spatial dims
            cond_latents = cond_latents[:, :t_lat, :h_lat, :w_lat, :]
            mx.eval(cond_latents)

            # Frame 0 is conditioned
            vision_condition_mask = vision_condition_mask.at[0, 0, 0].add(mx.array(1.0))

            # Mix: conditioned frames get encoded latent, rest get noise
            # cond_latents is [1, T_lat, H_lat, W_lat, z_dim], mask broadcasts over spatial+channel
            mask_5d = vision_condition_mask.reshape(1, t_lat, 1, 1, 1)
            latents = mask_5d * cond_latents + (1.0 - mask_5d) * latents
            mx.eval(latents)

        num_patches = t_lat * h_p * w_p

        # 2b. Prepare audio noise latents if enabled
        sound_latents = None
        sound_len = 0
        if enable_audio and num_frames <= 1:
            print("  Warning: --enable-audio ignored for single-frame generation")
        if enable_audio and num_frames > 1:
            sound_dim = 64  # Cosmos3 audio latent dim
            sampling_rate = 48000
            fps = 24
            hop_size = 1920
            n_audio_samples = int(num_frames / fps * sampling_rate)
            sound_len = (n_audio_samples + hop_size - 1) // hop_size
            sound_latents = mx.random.normal((1, sound_len, sound_dim)).astype(dtype)
            print(f"  Audio latents: {sound_len} frames ({n_audio_samples} samples @ {sampling_rate}Hz)")

        # 3. Set up schedulers
        self.scheduler.set_timesteps(num_inference_steps)
        if sound_latents is not None:
            audio_scheduler = UniPCScheduler()
            audio_scheduler.set_timesteps(num_inference_steps)

        print(f"  Latent shape: ({t_lat}, {h_lat}, {w_lat}, {z_dim})")
        print(f"  Patches: {num_patches} ({t_lat}×{h_p}×{w_p})")
        print(f"  Denoising steps: {num_inference_steps}")

        # 4. Denoising loop with text KV caching
        # Step 0: full forward (both pathways), cache understanding K/V
        # Steps 1+: generation pathway only, reuse cached K/V
        cond_kv_cache = None
        uncond_kv_cache = None
        cond_position_ids = None
        uncond_position_ids = None

        # Compute noisy frame indexes for selective timestep embedding (i2v).
        # For t2v (no conditioning): None = all frames get timestep.
        # For i2v: exclude conditioned frames (frame 0).
        noisy_fi = None
        if has_image_condition:
            cond_frame_set = set()
            for fi in range(t_lat):
                if vision_condition_mask[fi, 0, 0].item() > 0:
                    cond_frame_set.add(fi)
            noisy_fi = [fi for fi in range(t_lat) if fi not in cond_frame_set]

        for i in range(num_inference_steps):
            sigma = float(self.scheduler.sigmas[i].item())

            # Patchify current latents
            gen_tokens = self._patchify_latents(latents).astype(dtype)

            # Timestep tensor (sigma * num_train_timesteps)
            t_tensor = mx.array([sigma * self.scheduler.num_train_timesteps]).astype(dtype)

            # Audio tokens for this step
            audio_tokens = sound_latents if sound_latents is not None else None

            if cond_kv_cache is None:
                # Step 0: full forward, build cache
                cond_result, cond_kv_cache = self.model.diffusion_forward(
                    cond_ids, gen_tokens, t_tensor,
                    grid_t=t_lat, grid_h=h_p, grid_w=w_p,
                    audio_tokens=audio_tokens,
                    noisy_frame_indexes=noisy_fi,
                )
                cond_text_len = cond_ids.shape[1]
                cond_position_ids = self._build_position_ids(
                    cond_text_len, t_lat, h_p, w_p, audio_tokens,
                )
                mx.eval(*[kv[0] for kv in cond_kv_cache], *[kv[1] for kv in cond_kv_cache])
            else:
                # Steps 1+: cached forward (generation pathway only)
                cond_result = self.model.diffusion_forward_cached(
                    gen_tokens, t_tensor, cond_kv_cache,
                    cond_position_ids, cond_ids.shape[1],
                    audio_tokens=audio_tokens,
                    grid_t=t_lat, grid_h=h_p, grid_w=w_p,
                    noisy_frame_indexes=noisy_fi,
                )

            # Classifier-free guidance
            if guidance_scale != 1.0:
                if audio_tokens is not None:
                    cond_velocity, cond_audio_vel = cond_result
                    mx.eval(cond_velocity, cond_audio_vel)
                else:
                    cond_velocity = cond_result
                    mx.eval(cond_velocity)

                if uncond_kv_cache is None:
                    # Step 0: full uncond forward, build cache
                    uncond_result, uncond_kv_cache = self.model.diffusion_forward(
                        uncond_ids, gen_tokens, t_tensor,
                        grid_t=t_lat, grid_h=h_p, grid_w=w_p,
                        audio_tokens=audio_tokens,
                        noisy_frame_indexes=noisy_fi,
                    )
                    uncond_text_len = uncond_ids.shape[1]
                    uncond_position_ids = self._build_position_ids(
                        uncond_text_len, t_lat, h_p, w_p, audio_tokens,
                    )
                    mx.eval(*[kv[0] for kv in uncond_kv_cache], *[kv[1] for kv in uncond_kv_cache])
                else:
                    uncond_result = self.model.diffusion_forward_cached(
                        gen_tokens, t_tensor, uncond_kv_cache,
                        uncond_position_ids, uncond_ids.shape[1],
                        audio_tokens=audio_tokens,
                        noisy_frame_indexes=noisy_fi,
                        grid_t=t_lat, grid_h=h_p, grid_w=w_p,
                    )

                if audio_tokens is not None:
                    uncond_velocity, uncond_audio_vel = uncond_result
                    velocity_patches = uncond_velocity + guidance_scale * (
                        cond_velocity - uncond_velocity
                    )
                    audio_vel = uncond_audio_vel + guidance_scale * (
                        cond_audio_vel - uncond_audio_vel
                    )
                else:
                    uncond_velocity = uncond_result
                    velocity_patches = uncond_velocity + guidance_scale * (
                        cond_velocity - uncond_velocity
                    )
            else:
                if audio_tokens is not None:
                    velocity_patches, audio_vel = cond_result
                else:
                    velocity_patches = cond_result

            # Unpatchify velocity back to latent shape, crop to original dims
            velocity = self._unpatchify_latents(
                velocity_patches, t_lat, h_p, w_p,
                h_orig=h_lat, w_orig=w_lat,
            )

            # Zero velocity at conditioned frame positions (i2v).
            # With flow-matching: x_{t-1} = x_t + step * velocity.
            # Zeroing velocity keeps conditioned frames fixed at their initial value.
            if has_image_condition:
                mask_5d = vision_condition_mask.reshape(1, t_lat, 1, 1, 1)
                velocity = velocity * (1.0 - mask_5d)

            # Scheduler step (uses internal step_index)
            latents = self.scheduler.step(velocity, t_tensor, latents)
            mx.eval(latents)

            # Audio scheduler step
            if sound_latents is not None:
                # audio_vel: [1, sound_len, sound_dim] — already in latent shape
                sound_latents = audio_scheduler.step(audio_vel, t_tensor, sound_latents)
                mx.eval(sound_latents)

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Step {i+1}/{num_inference_steps} (σ={sigma:.4f})")

        # 5. Return normalized latents (decode_latents handles denormalization)
        result = {"latents": latents}

        if sound_latents is not None:
            # Store audio latents as [sound_dim, T] channels-first for audio decoder
            result["audio_latents"] = mx.transpose(sound_latents[0], (1, 0))

        print("  Done!")
        return result


def save_video(
    video_frames: np.ndarray,
    output_path: str,
    fps: int = 24,
    audio_waveform: np.ndarray = None,
    audio_sample_rate: int = 48000,
) -> str:
    """Save video frames (and optional audio) as MP4 or GIF.

    Args:
        video_frames: [T, H, W, 3] uint8 frames
        output_path: output file path (.mp4 or .gif)
        fps: video frame rate
        audio_waveform: optional [2, N] or [N] float audio in [-1, 1]
        audio_sample_rate: audio sample rate in Hz

    Returns:
        output path
    """
    import shutil

    output_path = str(output_path)
    is_mp4 = output_path.endswith(".mp4")

    if is_mp4 and shutil.which("ffmpeg") is None:
        if audio_waveform is not None:
            print("Warning: ffmpeg not found. Audio will be dropped. Saving as GIF.")
        else:
            print("Warning: ffmpeg not found. Saving as GIF instead.")
        output_path = output_path.rsplit(".", 1)[0] + ".gif"
        is_mp4 = False

    if is_mp4 and audio_waveform is not None:
        # Write frames as PNG sequence + WAV, mux with ffmpeg
        with tempfile.TemporaryDirectory() as tmpdir:
            from PIL import Image

            # Write frames
            for i, frame in enumerate(video_frames):
                Image.fromarray(frame).save(f"{tmpdir}/frame_{i:04d}.png")

            # Write WAV
            wav_path = f"{tmpdir}/audio.wav"
            if audio_waveform.ndim == 1:
                audio_waveform = audio_waveform[np.newaxis, :]
            n_channels = audio_waveform.shape[0]
            audio_int16 = (audio_waveform * 32767).clip(-32768, 32767).astype(np.int16)
            with wave.open(wav_path, "w") as wf:
                wf.setnchannels(n_channels)
                wf.setsampwidth(2)
                wf.setframerate(audio_sample_rate)
                if n_channels > 1:
                    interleaved = np.stack(
                        [audio_int16[c] for c in range(n_channels)], axis=-1
                    ).flatten()
                else:
                    interleaved = audio_int16[0]
                wf.writeframes(interleaved.tobytes())

            # Mux with ffmpeg
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", f"{tmpdir}/frame_%04d.png",
                "-i", wav_path,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                output_path,
            ]
            subprocess.run(cmd, check=True)

    elif is_mp4:
        # Video-only MP4
        with tempfile.TemporaryDirectory() as tmpdir:
            from PIL import Image

            for i, frame in enumerate(video_frames):
                Image.fromarray(frame).save(f"{tmpdir}/frame_{i:04d}.png")

            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", f"{tmpdir}/frame_%04d.png",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                output_path,
            ]
            subprocess.run(cmd, check=True)

    else:
        # GIF fallback
        from PIL import Image

        frames = [Image.fromarray(f) for f in video_frames]
        duration = int(1000 / fps)
        frames[0].save(
            output_path, save_all=True, append_images=frames[1:],
            duration=duration, loop=0,
        )

    return output_path
