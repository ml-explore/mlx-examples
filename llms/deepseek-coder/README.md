# Deepseek Coder

Deepseek Coder is an advanced series of code language models based on LLama architecture, trained from scratch on a massive corpus of 2T tokens, with a unique composition of 87% code and 13% natural language in both English and Chinese.

### Setup

Install the dependencies:

```
pip install -r requirements.txt
```

Next, download and convert the model. 
```sh
python convert.py --model-path <path_to_huggingface_model> --mlx-path <path_to_save_converted_model>
```
To generate a 4-bit quantized model, use -q. For a full list of options:

```
python convert.py --help
```

This process retrieves the model from Hugging Face. The default model is deepseek-ai/deepseek-coder-6.7b-instruct. Check out the [Hugging Face page](https://huggingface.co/deepseek-ai) to see a list of available models.

By default, the conversion script will save 
the converted `weights.npz`, `tokenizer`, and `config.json` there in the mlx-path you speficied .


### Run

Once you've converted the weights to MLX format, you can interact with the
Deepseek coder model:

```
python deepseek-coder.py  --model-path  <path_to_save_converted_model> --prompt "write a quick sort algorithm in python."
```

