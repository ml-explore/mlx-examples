import math
import time
from typing import Tuple

import mlx.core as mx
from tqdm import tqdm

from .utils import (
    load_ae,
    load_clip,
    load_clip_tokenizer,
    load_flow_model,
    load_t5,
    load_t5_tokenizer,
)


class FluxPipeline:
    def __init__(self, name: str):
        self.name = name
        self.ae = load_ae(name)
        self.flow = load_flow_model(name)
        self.clip = load_clip(name)
        self.clip_tokenizer = load_clip_tokenizer(name)
        self.t5 = load_t5(name)
        self.t5_tokenizer = load_t5_tokenizer(name)
        self.dtype = mx.bfloat16

    def ensure_models_are_loaded(self):
        mx.eval(
            self.ae.parameters(),
            self.flow.parameters(),
            self.clip.parameters(),
            self.t5.parameters(),
        )

    def _prior(self, n_images: int = 1, latent_size: Tuple[int, int] = (64, 64)):
        return mx.random.normal(
            shape=(n_images, *latent_size, 16),
            dtype=self.dtype,
        )

    def _prepare(self, x, text):
        b, h, w, c = x.shape

        # Prepare the latent image input and its ids for positional encoding
        x = x.reshape(b, h // 2, 2, w // 2, 2, c)
        x = x.transpose(0, 1, 3, 5, 2, 4).reshape(b, h * w // 4, c * 4)
        x_ids = mx.concatenate(
            [
                mx.zeros((h // 2, w // 2, 1), dtype=mx.int32),
                mx.broadcast_to(mx.arange(h // 2)[:, None, None], (h // 2, w // 2, 1)),
                mx.broadcast_to(mx.arange(w // 2)[None, :, None], (h // 2, w // 2, 1)),
            ],
            axis=-1,
        )
        x_ids = mx.broadcast_to(x_ids.reshape(1, h * w // 4, 3), (b, h * w // 4, 3))

        # Prepare the text features
        t5_tokens = mx.array([self.t5_tokenizer.tokenize(text)])
        txt = self.t5(t5_tokens)
        txt = mx.broadcast_to(txt, (b, *txt.shape[1:]))
        txt_ids = mx.zeros((b, txt.shape[1], 3), dtype=mx.int32)

        # Prepare the clip text features
        clip_tokens = mx.array([self.clip_tokenizer.tokenize(text)])
        vec = self.clip(clip_tokens).pooled_output
        vec = mx.broadcast_to(vec, (b, *vec.shape[1:]))

        return {
            "img": x,
            "img_ids": x_ids,
            "txt": txt,
            "txt_ids": txt_ids,
            "vec": vec,
        }

    def _get_shedule(
        self,
        num_steps,
        image_seq_len,
        base_shift: float = 0.5,
        max_shift: float = 1.5,
        shift: bool = True,
    ):
        timesteps = mx.linspace(1, 0, num_steps + 1)

        if shift:
            x = image_seq_len
            x1, x2 = 256, 4096
            y1, y2 = base_shift, max_shift
            mu = (x - x1) * (y2 - y1) / (x2 - x1) + y1
            timesteps = math.exp(mu) / (math.exp(mu) + (1 / timesteps - 1))

        return timesteps

    def generate_latents(
        self,
        text: str,
        n_images: int = 1,
        num_steps: int = 35,
        guidance: float = 4.0,
        latent_size: Tuple[int, int] = (64, 64),
        seed=None,
    ):
        # Set the PRNG state
        seed = int(time.time()) if seed is None else seed
        mx.random.seed(seed)

        # Create the latent variables
        x_T = self._prior(n_images, latent_size)

        # Get the initial inputs
        inputs = self._prepare(x_T, text)

        # Perform the denoising loop
        mx.eval(inputs)
        timesteps = self._get_shedule(
            num_steps, x_T.shape[1], shift="schnell" not in self.name
        )
        timesteps = timesteps.tolist()
        guidance = mx.full((n_images,), guidance, dtype=self.dtype)
        for t, t_prev in tqdm(zip(timesteps[:-1], timesteps[1:])):
            t_arr = mx.full((n_images,), t, dtype=self.dtype)
            pred = self.flow(
                img=inputs["img"],
                img_ids=inputs["img_ids"],
                txt=inputs["txt"],
                txt_ids=inputs["txt_ids"],
                y=inputs["vec"],
                timesteps=t_arr,
                guidance=guidance,
            )

            inputs["img"] = inputs["img"] + (t_prev - t) * pred
            mx.eval(inputs["img"])

        img = inputs["img"]
        h, w = latent_size
        img = img.reshape(n_images, h // 2, w // 2, -1, 2, 2)
        img = img.transpose(0, 1, 4, 2, 5, 3).reshape(n_images, h, w, -1)
        img = self.ae.decode(img)
        mx.eval(img)

        return ((mx.clip(img, -1, 1) + 1) * 127.5).astype(mx.uint8)
