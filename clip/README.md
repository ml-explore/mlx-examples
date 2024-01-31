# CLIP

An example of OpenAI's CLIP in MLX. The Contrastive Language-Image Pre-training (CLIP)
model embeds images and text in the same space.[^1]

### Setup

Install the dependencies:

```
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
from clip import load

model, tokenizer, img_processor = load("mlx_model")
inputs = {
    "input_ids": tokenizer(["a photo of a cat", "a photo of a dog"]),
    "pixel_values": img_processor(
        [Image.open("assets/cat.jpeg"), Image.open("assets/dog.jpeg")]
    ),
}
output = model(**inputs)
```

You can embed only images or only the text by passing just the
``input_ids`` or ``pixel_values``, respectively.

It is also possible to use ``transformers`` preprocessing utilities.
This is demonstrated in `example.py`:
```
python example.py
```

This example has been tested on the following Hugging Face repos:

- [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)
- [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)

You can run the test with:
```
python test.py
```

This compares the MLX implementation to the Transformers PyTorch
implementation. You can test new models by updating the `TEST_CKPT` list in
`test.py`.

### Attribution

- *"assets/cat.jpeg"* is a "Cat" by London's, licensed under CC BY-SA 2.0.
- *"assets/dog.jpeg"* is a "Happy Dog" by tedmurphy, licensed under CC BY 2.0.

[^1]: Refer to the original paper [Learning Transferable Visual Models From
  Natural Language Supervision ](https://arxiv.org/abs/2103.00020) or [blog
  post](https://openai.com/research/clip)
