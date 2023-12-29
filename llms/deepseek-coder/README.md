# Deepseek Coder

Deepseek Coder is a family of code generating language models based on the
LLama architecture.[^1] The models were trained from scratch on a corpus of 2T
tokens, with a composition of 87% code and 13% natural language containing both
English and Chinese.

### Setup

Install the dependencies:

```
pip install -r requirements.txt
```

Next, download and convert the model. 

```sh
python convert.py --hf-path <path_to_huggingface_model> --mlx-path <path_to_save_converted_model>
```

To generate a 4-bit quantized model, use `-q`. For a full list of options run:

```
python convert.py --help
```

The converter downloads the model from Hugging Face. The default model is
`deepseek-ai/deepseek-coder-6.7b-instruct`. Check out the Hugging Face
page[^1] to see a list of available models.

By default, the conversion script will save the converted `weights.npz`,
`tokenizer`, and `config.json` in the path provided by `--mlx-path`.


### Run

Once you've converted the weights to MLX format, you can interact with the
Deepseek coder model:

```
python deepseek-coder.py  --model-path  <path_to_save_converted_model> --prompt "write a quick sort algorithm in python."
```

[^1] For more information see the [Hugging Face page](https://huggingface.co/deepseek-ai).
