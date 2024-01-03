## Generate Text with MLX and :hugs: Hugging Face

This an example large language model text generation that can pull models from
the Hugging Face Hub.

### Setup

Install the dependencies:

```
pip install -r requirements.txt
```

### Run

```
python generate.py --model <model_path> --prompt "hello"
```

For example:

```
python generate.py --model mistralai/Mistral-7B-v0.1 --prompt "hello"
```

will download the Mistral 7B model and generate text using the given prompt.

The `<model_path>` should be either a path to a local directory or a Hugging
Face repo with weights stored in `safetensors` format. If you use a repo from
the Hugging Face hub, then the model will be downloaded and cached the first
time you run it. See the [Models](#models) section for a full list of supported models.

Run `python generate.py --help` to see all the options.


### Models

The example supports Hugging Face format Mistral and Llama-style models.  If the
model you want to run is not supported, file an
[issue](https://github.com/ml-explore/mlx-examples/issues/new) or better yet,
submit a pull request.

Here is a list of a few more Hugging Face models repos which work with this example:

- [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- [TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T)

Most
[Mistral](https://huggingface.co/models?library=transformers,safetensors&other=mistral&sort=trending)
and
[Llama](https://huggingface.co/models?library=transformers,safetensors&other=llama&sort=trending)
style models should work out of the box.

### Convert new models 

You can convert new models to the MLX format using the `convert.py` script.
This script takes a Hugging Face repo as input and outputs an MLX formatted
model (which you can then upload to Hugging Face).

To convert a model, run:

```
python convert.py --hf-model <hf_repo>
```

To make a 4-bit quantized model, use `-q`. For more options run:

```
python convert.py
```

You can upload new models to the [Hugging Face MLX
Community](https://huggingface.co/mlx-community) by specifying
`--upload-name`` to `convert.py`.


