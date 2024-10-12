import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image
from tqdm import tqdm


class Dataset:
    def __init__(self, flux, args):
        self.args = args
        self.flux = flux
        self.dataset_base = Path(args.dataset)
        data_file = self.dataset_base / "train.jsonl"
        if not data_file.exists():
            raise ValueError(f"The fine-tuning dataset 'train.jsonl' was not found in the '{args.dataset}' path.")
        with open(data_file, "r") as fid:
            self.data = [json.loads(l) for l in fid]

        self.latents = []
        self.t5_features = []
        self.clip_features = []

    def _random_crop_resize(self, img):
        resolution = self.args.resolution
        width, height = img.size

        a, b, c, d = mx.random.uniform(shape=(4,), stream=mx.cpu).tolist()

        # Random crop the input image between 0.8 to 1.0 of its original dimensions
        crop_size = (
            max((0.8 + 0.2 * a) * width, resolution[0]),
            max((0.8 + 0.2 * a) * height, resolution[1]),
        )
        pan = (width - crop_size[0], height - crop_size[1])
        img = img.crop(
            (
                pan[0] * b,
                pan[1] * c,
                crop_size[0] + pan[0] * b,
                crop_size[1] + pan[1] * c,
            )
        )

        # Fit the largest rectangle with the ratio of resolution in the image
        # rectangle.
        width, height = crop_size
        ratio = resolution[0] / resolution[1]
        r1 = (height * ratio, height)
        r2 = (width, width / ratio)
        r = r1 if r1[0] <= width else r2
        img = img.crop(
            (
                (width - r[0]) / 2,
                (height - r[1]) / 2,
                (width + r[0]) / 2,
                (height + r[1]) / 2,
            )
        )

        # Finally resize the image to resolution
        img = img.resize(resolution, Image.LANCZOS)

        return mx.array(np.array(img))

    def encode_images(self):
        """Encode the images in the latent space to prepare for training."""
        self.flux.ae.eval()
        for sample in tqdm(self.data, desc="encode images"):
            input_img = Image.open(self.dataset_base / sample["image"])
            for i in range(self.args.num_augmentations):
                img = self._random_crop_resize(input_img)
                img = (img[:, :, :3].astype(self.flux.dtype) / 255) * 2 - 1
                x_0 = self.flux.ae.encode(img[None])
                x_0 = x_0.astype(self.flux.dtype)
                mx.eval(x_0)
                self.latents.append(x_0)

    def encode_prompts(self):
        """Pre-encode the prompts so that we don't recompute them during
        training (doesn't allow finetuning the text encoders)."""
        for sample in tqdm(self.data, desc="encode prompts"):
            t5_tok, clip_tok = self.flux.tokenize([sample["prompt"]])
            t5_feat = self.flux.t5(t5_tok)
            clip_feat = self.flux.clip(clip_tok).pooled_output
            mx.eval(t5_feat, clip_feat)
            self.t5_features.append(t5_feat)
            self.clip_features.append(clip_feat)

    def iterate(self, batch_size):
        xs = mx.concatenate(self.latents)
        t5 = mx.concatenate(self.t5_features)
        clip = mx.concatenate(self.clip_features)
        mx.eval(xs, t5, clip)
        n_aug = self.args.num_augmentations
        while True:
            x_indices = mx.random.permutation(len(self.latents))
            c_indices = x_indices // n_aug
            for i in range(0, len(self.latents), batch_size):
                x_i = x_indices[i: i + batch_size]
                c_i = c_indices[i: i + batch_size]
                yield xs[x_i], t5[c_i], clip[c_i]


def load_dataset(flux, args):
    return Dataset(flux, args)
