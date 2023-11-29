# LLaMA

An example of generating text with LLaMA using MLX.

LLaMA is a set of open source language models from Meta AI Research[^1] ranging from 7B to 65B parameters.

### Setup

Install the dependencies:

```
pip install -r requirements.txt
```

Next, download and convert the model. If you do not have access to the model
weights you will need to [request
access](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform)
from Meta.

Convert the weights with:

```
python convert.py <path_to_torch_weights> mlx_llama_weights.npz
```

### Run

Once you've converted the weights to MLX format, you can interact with the
LLaMA model:

```
python llama.py mlx_llama.npz tokenizer.model "hello"
```

Run `python llama.py --help` for more details.

[^1]: Refer to the [arXiv paper](https://arxiv.org/abs/2302.13971) and [blog post](https://ai.meta.com/blog/large-language-model-llama-meta-ai/) for more details.
