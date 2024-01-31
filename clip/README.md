# CLIP

An example of OpenAI's CLIP in MLX. The CLIP (contrastive language-image
pre-training) model embeds images and text in the same space.[^1]

### Setup

Install the dependencies:

```shell
pip install -r requirements.txt
```

Next, download a CLIP model from Hugging Face and convert it to MLX. The
default model is
[openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32).

```
python convert.py
```

The script will by default download the model and configuration files to the
directory ``mlx_model/``.

### Run

You can use the CLIP model to embed images and text. 

```python
from PIL import Image
import clip

model, tokenizer, img_processor = clip.load("mlx_model")
inputs = {
    "input_ids": tokenizer(["a photo of a cat", "a photo of a dog"]),
    "pixel_values": img_processor(
        [Image.open("assets/cat.jpeg"), Image.open("assets/dog.jpeg")]
    ),
}
output = model(**inputs)

# Get text and image embeddings:
text_embeds = output.text_embeds
image_embeds = output.image_embeds
```

Run the above example with `python clip.py`.

To embed only images or only the text, pass only the ``input_ids`` or
``pixel_values``, respectively.

This example re-implements minimal image preprocessing and tokenization to reduce
dependencies. For additional preprocessing functionality, you can use
``transformers``. The file `hf_preproc.py` has an example.

MLX CLIP has been tested and works with the following Hugging Face repos:

- [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)
- [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)

You can run the tests with:

```shell
python test.py
```

To test new models, update the `MLX_PATH` and `HF_PATH` in `test.py`.

### Attribution

- `assets/cat.jpeg` is a "Cat" by London's, licensed under CC BY-SA 2.0.
- `assets/dog.jpeg` is a "Happy Dog" by tedmurphy, licensed under CC BY 2.0.

[^1]: Refer to the original paper [Learning Transferable Visual Models From
  Natural Language Supervision ](https://arxiv.org/abs/2103.00020) or [blog
  post](https://openai.com/research/clip)
