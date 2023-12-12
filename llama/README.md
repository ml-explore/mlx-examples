# Llama

An example of generating text with Llama (1 or 2) using MLX.

Llama is a set of open source language models from Meta AI Research[^1][^2]
ranging from 7B to 70B parameters.

### Setup

Install the dependencies:

```
pip install -r requirements.txt
```

Next, download and convert the model. If you do not have access to the model
weights you will need to request access from Meta:

- [Request Llama v1](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform)
- [Request Llama v2](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)


Alternatively, you can also download a select converted checkpoints from the
[mlx-llama](https://huggingface.co/mlx-llama) community organisation on Hugging
Face and skip the conversion step.

Convert the weights with:

```
python convert.py --model_path <path_to_torch_model>
```

The conversion script will save the converted weights in the same location.

### Run

Once you've converted the weights to MLX format, you can interact with the
LlaMA model:

```
python llama.py <path_to_model> <path_to_tokenizer.model> "hello"
```

Run `python llama.py --help` for more details.

[^1]: For Llama v1 refer to the [arXiv paper](https://arxiv.org/abs/2302.13971) and [blog post](https://ai.meta.com/blog/large-language-model-llama-meta-ai/) for more details.
[^2]: For Llama v2 refer to the [blob post](https://ai.meta.com/llama/)
