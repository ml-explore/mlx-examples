## Generate Text with LLMs and MLX

The easiest way to get started is to install the `mlx-lm` package:

**With `pip`**:

```sh
pip install mlx-lm
```

**With `conda`**:

```sh
conda install -c conda-forge mlx-lm
```

The `mlx-lm` package also has:

- [LoRA and QLoRA fine-tuning](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md)
- [Merging models](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/MERGE.md)
- [HTTP model serving](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/SERVER.md)

### Python API

You can use `mlx-lm` as a module:

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")

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

repo = "mistralai/Mistral-7B-Instruct-v0.3"
upload_repo = "mlx-community/My-Mistral-7B-Instruct-v0.3-4bit"

convert(repo, quantize=True, upload_repo=upload_repo)
```

This will generate a 4-bit quantized Mistral 7B and upload it to the repo
`mlx-community/My-Mistral-7B-Instruct-v0.3-4bit`. It will also save the
converted model in the path `mlx_model` by default.

To see a description of all the arguments you can do:

```
>>> help(convert)
```

#### Streaming

For streaming generation, use the `stream_generate` function. This returns a
generator object which streams the output text. For example,

```python
from mlx_lm import load, stream_generate

repo = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
model, tokenizer = load(repo)

prompt = "Write a story about Einstein"

for t in stream_generate(model, tokenizer, prompt, max_tokens=512):
    print(t, end="", flush=True)
print()
```

### Command Line

You can also use `mlx-lm` from the command line with:

```
mlx_lm.generate --model mistralai/Mistral-7B-Instruct-v0.3 --prompt "hello"
```

This will download a Mistral 7B model from the Hugging Face Hub and generate
text using the given prompt.

For a full list of options run:

```
mlx_lm.generate --help
```

To quantize a model from the command line run:

```
mlx_lm.convert --hf-path mistralai/Mistral-7B-Instruct-v0.3 -q
```

For more options run:

```
mlx_lm.convert --help
```

You can upload new models to Hugging Face by specifying `--upload-repo` to
`convert`. For example, to upload a quantized Mistral-7B model to the
[MLX Hugging Face community](https://huggingface.co/mlx-community) you can do:

```
mlx_lm.convert \
    --hf-path mistralai/Mistral-7B-Instruct-v0.3 \
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
- [Qwen/Qwen-7B](https://huggingface.co/Qwen/Qwen-7B)
- [pfnet/plamo-13b](https://huggingface.co/pfnet/plamo-13b)
- [pfnet/plamo-13b-instruct](https://huggingface.co/pfnet/plamo-13b-instruct)
- [stabilityai/stablelm-2-zephyr-1_6b](https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b)
- [internlm/internlm2-7b](https://huggingface.co/internlm/internlm2-7b)

Most
[Mistral](https://huggingface.co/models?library=transformers,safetensors&other=mistral&sort=trending),
[Llama](https://huggingface.co/models?library=transformers,safetensors&other=llama&sort=trending),
[Phi-2](https://huggingface.co/models?library=transformers,safetensors&other=phi&sort=trending),
and
[Mixtral](https://huggingface.co/models?library=transformers,safetensors&other=mixtral&sort=trending)
style models should work out of the box.

For some models (such as `Qwen` and `plamo`) the tokenizer requires you to
enable the `trust_remote_code` option. You can do this by passing
`--trust-remote-code` in the command line. If you don't specify the flag
explicitly, you will be prompted to trust remote code in the terminal when
running the model. 

For `Qwen` models you must also specify the `eos_token`. You can do this by
passing `--eos-token "<|endoftext|>"` in the command
line. 

These options can also be set in the Python API. For example:

```python
model, tokenizer = load(
    "qwen/Qwen-7B",
    tokenizer_config={"eos_token": "<|endoftext|>", "trust_remote_code": True},
)
```
