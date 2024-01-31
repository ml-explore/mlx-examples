# (OpenAI) CLIP (ViT-based)

An example of visual-language representation learning using MLX.

CLIP, for Contrastive Language-Image Pre-training, is a powerful representation learning method that aims to learn shared embedding space for both images and text.

Paper: https://arxiv.org/pdf/2103.00020.pdf

### Setup

Install the dependencies:

```
pip install -r requirements.txt
```

Next, download the model configuration and weights from HuggingFace and convert the weights to `mlx`. 
In this example, we will use ```openai/clip-vit-base-patch32```.

```
python convert.py --hf-repo openai/clip-vit-base-patch32 --mlx-path weights/openai/clip-vit-base-patch32
```
If the download and conversion are successful, the folder ```weights``` will appear. In the ```weights``` folder, there will be the HuggingFace configuration with 
the converted weights for ```openai/clip-vit-base-patch32```.

### Run

Once you've downloaded and converted the weights to `mlx`, you can use the
CLIP model to embed images and text. 

```python
from clip import load
from PIL import Image

MODEL: str = "openai/clip-vit-base-patch32"
CONVERTED_CKPT_PATH: str = f"weights/{MODEL}"

# Load MLX CLIP model, tokenizer and image (pre)processor
(
    mlx_clip,
    tokenizer,
    img_processor
) = load(CONVERTED_CKPT_PATH)

# Preprocess the input
clip_input = {
    "input_ids": tokenizer(["a photo of a cat", "a photo of a dog"]),
    "pixel_values": img_processor([Image.open("assets/cat.jpeg"), Image.open("assets/dog.jpeg")])
}
# Compute the output
mlx_out = mlx_clip(
    **clip_input,
    return_loss=True
)
# Print some embeddings and the CLIP loss
print("text embeddings:")
print(mlx_out.text_embeds)
print("image embeddings:")
print(mlx_out.image_embeds)
print(f"CLIP loss: {mlx_out.loss}")
```
The output should be:
```
text embeddings:
array([[0.0148391, 0.0069961, -0.0233705, ..., -0.0508463, -0.0437914, 0.00330403],
       [0.00870739, 0.0258293, -0.0386577, ..., -0.0546769, -0.0241999, 0.0111514]], dtype=float32)
image embeddings:
array([[-0.000217405, -0.00493075, 0.0141711, ..., 0.0798553, -0.0224953, -0.0192719],
       [0.000887729, -0.0116987, -0.0106347, ..., 0.0521465, -0.00254958, -0.0034469]], dtype=float32)
CLIP loss: array(0.00633574, dtype=float32)
```

It is also possible to embed only the images or only the text.
Thus, both ``input_ids`` and ``pixel_values`` parameters are optional. 
To embed only the text, simply provide only the `input_ids`. Similarly, to embed only the images, simply provide only the ``pixel_values``.

It is also possible to use ``transformers`` preprocessing utilities.
This is demonstrated in `example.py`:
```
python example.py
```

### Remarks
The conversion method and the correctness of the CLIP implementation were tested on the following HuggingFace repos:
- `openai/clip-vit-base-patch32`
- `openai/clip-vit-large-patch14`

To verify the correctness of the CLIP implementation by comparing it to `transformers` PyTorch implementation, adapt `test.py` (e.g. choose the desired testing checkpoint by setting `TEST_CKPTS`) and run:
```
python test.py
```

### Photo Attribution
- *"assets/cat.jpeg"* is a "Cat" by London's, licensed under CC BY-SA 2.0.
- *"assets/dog.jpeg"* is a "Happy Dog" by tedmurphy, licensed under CC BY 2.0.