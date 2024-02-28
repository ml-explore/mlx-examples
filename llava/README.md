# LLaVA

An example of LLaVA: Large Language and Vission Assistant in MLX. LLlava is a
multi-modal model that can generate text from images and text prompts.[^1]

## Setup:

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run

You can use LlaVA model to ask questions about images.

For example using the command line:

```bash
python generate.py --model_path llava-hf/llava-1.5-7b-hf --image "http://images.cocodataset.org/val2017/000000039769.jpg" --prompt "USER: <image>\nWhat are these?\nASSISTANT:" --max_tokens 128 --temperature 0
```

Or directly in Python:

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
input_ids, pixel_values = prepare_inputs(processor, image, prompt)

reply = generate_text(input_ids, pixel_values, model, processor, max_tokens, temperature)

print(reply)
```

[^1]:
    Refer to [LLaVA project webpage](https://llava-vl.github.io/) for more
    information.
