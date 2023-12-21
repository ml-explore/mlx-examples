# Qwen

Qwen (通义千问) are a family of language models developed by Alibaba Cloud.[^1]
The architecture of the Qwen models is similar to Llama except for the bias in
the attention layers.

## Setup

First download and convert the model with: 

```sh
python convert.py
```

To generate a 4-bit quantized model, use ``-q``. For a full list of options:

The script downloads the model from Hugging Face. The default model is
`Qwen/Qwen-1_8B`. Check out the [Hugging Face
page](https://huggingface.co/Qwen) to see a list of available models.

By default, the conversion script will make the directory `mlx_model` and save
the converted `weights.npz` and `config.json` there.

## Generate

To generate text with the default prompt:

```sh
python qwen.py
```

If you change the model, make sure to pass the corresponding tokenizer. E.g.,
for Qwen 7B use:

```
python qwen.py --tokenizer  Qwen/Qwen-7B
```

To see a list of options, run:

```sh
python qwen.py --help
```

[^1]: For more details on the model see the official repo of [Qwen](https://github.com/QwenLM/Qwen) and the [Hugging Face](https://huggingface.co/Qwen).
