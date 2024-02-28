# LLaVA

An example of LLaVA: Large Language and Vission Assistant in MLX. LLlava is a multi-modal model that can generate text from images and text prompts. [^1]

## Setup:

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run

You can use LlaVA model to ask questions about images.

The python snippet below shows how to use the model to ask questions about an image.

```python
from llava import LlavaModel
from transformers import AutoProcessor
from utils import load_image, prepare_inputs
from generate import generate_text
model_path = 'llava-hf/llava-1.5-7b-hf'

processor = AutoProcessor.from_pretrained(model_path)
model = LlavaModel.from_pretrained(model_path)

max_tokens, temperature = 128, 0.

prompt = "USER: <image>\nWhat are these?\nASSISTANT:"
image = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(image)
input_ids, pixel_values = prepare_inputs(prompt, image, processor)

reply = generate_text(input_ids, pixel_values, model, processor, max_tokens, temperature)

print(reply)
```

[^1]: Please refer to original LlaVA library for more details: [https://github.com/haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA)
