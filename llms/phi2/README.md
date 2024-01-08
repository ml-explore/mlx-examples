# Phi-2

Phi-2 is a 2.7B parameter language model released by Microsoft with
performance that rivals much larger models.[^1] It was trained on a mixture of
GPT-4 outputs and clean web text.

Phi-2 efficiently runs on Apple silicon devices with 8GB of memory in 16-bit
precision.

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
python generate.py --model microsoft/phi-2 --prompt "hello"
```
The `<model_path>` should be either a path to a local directory or a Hugging
Face repo with weights stored in `safetensors` format. If you use a repo from
the Hugging Face Hub, then the model will be downloaded and cached the first
time you run it. 

Run `python generate.py --help` to see all the options.

### Convert new models 

You can convert (change the data type or quantize) models using the
`convert.py` script. This script takes a Hugging Face repo as input and outputs
a model directory (which you can optionally also upload to Hugging Face).

For example, to make 4-bit quantized a model, run:

```
python convert.py --hf-path <hf_repo> -q
```

For more options run:

```
python convert.py --help
```

You can upload new models to the [Hugging Face MLX
Community](https://huggingface.co/mlx-community) by specifying `--upload-name``
to `convert.py`.

[^1]: For more details on the model see the [blog post](
https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/)
and the [Hugging Face repo](https://huggingface.co/microsoft/phi-2)