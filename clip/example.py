from pathlib import Path

import mlx.core as mx
import model
import numpy as np
import torch
import transformers
from PIL import Image

MODEL: str = "openai/clip-vit-base-patch32"
CONVERTED_CKPT_PATH: str = f"weights/mlx/{MODEL}"

mlx_clip = model.CLIPModel.from_pretrained(Path(CONVERTED_CKPT_PATH))
tf_clip = transformers.CLIPModel.from_pretrained(MODEL)
tf_processor = transformers.CLIPProcessor.from_pretrained(MODEL)

clip_input = tf_processor(
    text=["a photo of a cat", "a photo of a dog"],
    images=[Image.open("cats.jpeg"), Image.open("dog.jpeg")],
    return_tensors="pt",
)

with torch.inference_mode():
    torch_out = tf_clip(**clip_input, return_loss=True)

mlx_out = mlx_clip(
    input_ids=mx.array(clip_input.input_ids.numpy()),
    pixel_values=mx.array(clip_input.pixel_values.numpy()).transpose((0, 2, 3, 1)),
    return_loss=True,
)

assert np.allclose(mlx_out.text_embeds, torch_out.text_embeds, atol=1e-5)
assert np.allclose(mlx_out.image_embeds, torch_out.image_embeds, atol=1e-5)
assert np.allclose(mlx_out.loss, torch_out.loss, atol=1e-5)

print("text embeddings:")
print(mlx_out.text_embeds)
print("image embeddings:")
print(mlx_out.image_embeds)
print(f"CLIP loss: {mlx_out.loss}")
