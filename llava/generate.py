import mlx.core as mx
import mlx.nn as nn
import requests
from PIL import Image
from transformers import AutoProcessor

from llava import LlavaModel

MODEL_PATH = "models/llava-hf/llava-1.5-7b-hf"

prompt = "USER: <image>\nWhat are these?\nASSISTANT:"
image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
raw_image = Image.open(requests.get(image_file, stream=True).raw)


processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = LlavaModel.from_pretrained(MODEL_PATH)

values = processor(prompt, raw_image, return_tensors="np")
pixel_values = mx.array(values["pixel_values"])
input_ids = mx.array(values["input_ids"])

input_embeds = model(input_ids, pixel_values)
max_tokens = 100
temperature = 0.3


def sample(logits, temp=0.0):
    if temp == 0:
        return mx.argmax(logits, axis=-1)
    else:
        return mx.random.categorical(logits * (1 / temp))


def generate(y: mx.array, model: nn.Module, temp: float = 0.0, cache=None):
    while True:
        logits, cache = model(y[None], cache=cache)
        logits = logits[:, -1, :]

        y = sample(logits, temp=temp)
        token = y.item()

        yield token


logits, cache = model.language_model(input_ids, cache=None, inputs_embeds=input_embeds)
logits = logits[:, -1, :]
y = sample(logits, temp=temperature)
tokens = [y.item()]
for token, _ in zip(
    generate(y, model.language_model, temperature, cache=cache),
    range(max_tokens),
):
    if token == processor.tokenizer.eos_token_id:
        break
    tokens.append(token)

print(processor.tokenizer.decode(tokens))
