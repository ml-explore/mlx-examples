## Generate Text in MLX

This an example of Llama style large language model text generation.

### Setup

Install the dependencies:

```
pip install -r requirements.txt
```

### Run

```
python generate.py --model <model_path> --prompt "hello"
```

The `<model_path>` should be either a path to a local directory with an MLX
formatted model, or a Hugging Face repo. If the latter, then the model will
be downloaded and cached the first time you use it. See the [#Models] section
for a full list of supported models.

Run `python generate.py --help` to see all the options.


### Models

The following models (and variants) are supported:

Hugging Face Repo | Model Size |
----------------- | ---------- |
N/A | N/A 

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

The conversion script supports Hugging Face format Llama-style models.  If the
model you want to convert is not supported, file an
[issue](https://github.com/ml-explore/mlx-examples/issues/new) or better yet,
submit a pull request.

This is a list of Hugging Face models which have been tested with the
conversion script:

- meta-llama/Llama-2-7b-hf
- mistralai/Mistral-7B-v0.1
- TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
