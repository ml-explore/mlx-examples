# Llama

An example of generating text with Llama (1 or 2) using MLX.

Llama is a set of open source language models from Meta AI Research[^1][^2]
ranging from 7B to 70B parameters. This example also supports Meta's Llama Chat
and Code Llama models, as well as the 1.1B TinyLlama models from SUTD.[^3]

### Setup

Install the dependencies:

```
pip install -r requirements.txt
```

Next, download and convert the model. If you do not have access to the model
weights you will need to request access from Meta:

- [Request Llama v1](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform)
- [Request Llama v2](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)

> [!TIP] Alternatively, you can also download a few converted checkpoints from
> the [MLX Community](https://huggingface.co/mlx-community) organization on
> Hugging Face and skip the conversion step.

You can download the TinyLlama models directly from [Hugging
Face](https://huggingface.co/TinyLlama).

Convert the weights with:

```
python convert.py --model-path <path_to_torch_model>
```

For TinyLlama use

```
python convert.py --model-path <path_to_torch_model> --model-name tiny_llama
```

The conversion script will save the converted weights in the same location.

### Run

Once you've converted the weights to MLX format, you can interact with the
LlaMA model:

```
python llama.py <path_to_model> <path_to_tokenizer.model> --prompt "hello"
```

Run `python llama.py --help` for more details.

[^1]: For Llama v1 refer to the [arXiv paper](https://arxiv.org/abs/2302.13971) and [blog post](https://ai.meta.com/blog/large-language-model-llama-meta-ai/) for more details.
[^2]: For Llama v2 refer to the [blob post](https://ai.meta.com/llama/)
[^3]: For TinyLlama refer to the [gihub repository](https://github.com/jzhang38/TinyLlama?tab=readme-ov-file)
