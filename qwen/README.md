# Qwen

Qwen (通义千问) is a language model proposed by Alibaba Cloud[^1]. The architecture of Qwen is similar to Llama except for the bias in the attention layers.

## Setup

Download (from huggingface) and conver the model. By default, the model is `Qwen/Qwen-1_8B`.

```sh
python convert.py
```

This will make the `weights.npz` file which MLX can read.

## Generate

To generate text with the default prompt (default tokenizer is `Qwen/Qwen-1_8B`):

```sh
python qwen.py
```

To see a list of options, run:

```sh
python qwen.py --help
```

[^1]: For more details on the model see the official repo of [Qwen](https://github.com/QwenLM/Qwen) and the [Hugging Face](https://huggingface.co/Qwen).
