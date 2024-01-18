## Generate Text with LLMs and MLX

The easiest way to get started is to install the `mlx-lm` package:

```shell
pip install mlx-lm
```

### Python API

You can use `mlx-lm` as a module:

```python
from mlx_lm import load, generate

model, tokenizer = load("mistralai/Mistral-7B-v0.1")

response = generate(model, tokenizer, prompt="hello", verbose=True)
```

To see a description of all the arguments you can do:

```
>>> help(generate)
```

The `mlx-lm` package also comes with functionality to quantize and optionally
upload models to the Hugging Face Hub.

You can convert models in the Python API with:

```python
from mlx_lm import convert 

upload_repo = "mlx-community/My-Mistral-7B-v0.1-4bit"

convert("mistralai/Mistral-7B-v0.1", quantize=True, upload_repo=upload_repo)
```

This will generate a 4-bit quantized Mistral-7B and upload it to the
repo `mlx-community/My-Mistral-7B-v0.1-4bit`. It will also save the
converted model in the path `mlx_model` by default.

To see a description of all the arguments you can do:

```
>>> help(convert)
```

### Command Line 

You can also use `mlx-lm` from the command line with:

```
python -m mlx_lm.generate --model mistralai/Mistral-7B-v0.1 --prompt "hello"
```

This will download a Mistral 7B model from the Hugging Face Hub and generate
text using the given prompt. 

For a full list of options run:

```
python -m mlx_lm.generate --help
```

To quantize a model from the command line run:

```
python -m mlx_lm.convert --hf-path mistralai/Mistral-7B-v0.1 -q 
```

For more options run:

```
python -m mlx_lm.convert --help
```

You can upload new models to Hugging Face by specifying `--upload-repo` to
`convert`. For example, to upload a quantized Mistral-7B model to the 
[MLX Hugging Face community](https://huggingface.co/mlx-community) you can do:

```
python -m mlx_lm.convert \
    --hf-path mistralai/Mistral-7B-v0.1 \
    -q \
    --upload-repo mlx-community/my-4bit-mistral
```

### Supported Models

The example supports Hugging Face format Mistral, Llama, and Phi-2 style
models.  If the model you want to run is not supported, file an
[issue](https://github.com/ml-explore/mlx-examples/issues/new) or better yet,
submit a pull request.

Here are a few examples of Hugging Face models that work with this example:

- [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- [deepseek-ai/deepseek-coder-6.7b-instruct](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct)
- [01-ai/Yi-6B-Chat](https://huggingface.co/01-ai/Yi-6B-Chat)
- [microsoft/phi-2](https://huggingface.co/microsoft/phi-2)
- [mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)

Most
[Mistral](https://huggingface.co/models?library=transformers,safetensors&other=mistral&sort=trending),
[Llama](https://huggingface.co/models?library=transformers,safetensors&other=llama&sort=trending),
[Phi-2](https://huggingface.co/models?library=transformers,safetensors&other=phi&sort=trending)
and
[Mixtral](https://huggingface.co/models?library=transformers,safetensors&other=mixtral&sort=trending)
style models should work out of the box.
