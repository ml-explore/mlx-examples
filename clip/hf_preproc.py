import mlx.core as mx
import transformers
from PIL import Image

import clip

hf_model = "openai/clip-vit-base-patch32"
mlx_model = "mlx_model"

model, *_ = clip.load(mlx_model)
processor = transformers.CLIPProcessor.from_pretrained(hf_model)

inputs = processor(
    text=["a photo of a cat", "a photo of a dog"],
    images=[Image.open("assets/cat.jpeg"), Image.open("assets/dog.jpeg")],
    return_tensors="np",
)

out = model(
    input_ids=mx.array(inputs.input_ids),
    pixel_values=mx.array(inputs.pixel_values).transpose((0, 2, 3, 1)),
    return_loss=True,
)

print("text embeddings:")
print(out.text_embeds)
print("image embeddings:")
print(out.image_embeds)
print(f"CLIP loss: {out.loss.item():.3f}")
