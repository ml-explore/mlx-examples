# LLaVA

An example of LLaVA: Large Language and Vision Assistant in MLX.[^1] LLlava is
a multimodal model that can generate text given combined image and text inputs.

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run

You can use LLaVA to ask questions about images.

For example, using the command line:

```bash
python generate.py \
  --model llava-hf/llava-1.5-7b-hf \
  --image "http://images.cocodataset.org/val2017/000000039769.jpg" \
  --prompt "USER: <image>\nWhat are these?\nASSISTANT:" \
  --max-tokens 128 \
  --temp 0
```

This uses the following image:

![alt text](http://images.cocodataset.org/val2017/000000039769.jpg)
 
And generates the output:

```
These are two cats lying on a pink couch.
```

You can also use LLaVA in Python:

```python
from generate import load_model, prepare_inputs, generate_text

processor, model = load_model("llava-hf/llava-1.5-7b-hf")

max_tokens, temperature = 128, 0.0

prompt = "USER: <image>\nWhat are these?\nASSISTANT:"
image = "http://images.cocodataset.org/val2017/000000039769.jpg"
input_ids, pixel_values = prepare_inputs(processor, image, prompt)

reply = generate_text(
    input_ids, pixel_values, model, processor, max_tokens, temperature
)

print(reply)
```

[^1]:
    Refer to [LLaVA project webpage](https://llava-vl.github.io/) for more
    information.
