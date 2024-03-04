# Whisper

Speech recognition with Whisper in MLX. Whisper is a set of open source speech
recognition models from OpenAI, ranging from 39 million to 1.5 billion
parameters.[^1]

### Setup

First, install the dependencies:

```
pip install -r requirements.txt
```

Install [`ffmpeg`](https://ffmpeg.org/):

```
# on macOS using Homebrew (https://brew.sh/)
brew install ffmpeg
```

> [!TIP]
> Skip the conversion step by using pre-converted checkpoints from the Hugging
> Face Hub. There are a few available in the [MLX
> Community](https://huggingface.co/mlx-community) organization.

To convert a model, first download the Whisper PyTorch checkpoint and convert
the weights to the MLX format. For example, to convert the `tiny` model use:

```
python convert.py --torch-name-or-path tiny --mlx-path mlx_models/tiny
```

Note you can also convert a local PyTorch checkpoint which is in the original OpenAI format.

To generate a 4-bit quantized model, use `-q`. For a full list of options:

```
python convert.py --help
```

By default, the conversion script will make the directory `mlx_models`
and save the converted `weights.npz` and `config.json` there. Note that
this does not automatically save different versions in this folder.
However, the downloaded models are cached for future use. 

Consider a scripted run (e.g. BASH, zsh) like the below example to stream-line filemaking:

 Here's a Python script that acts as a wrapper for calling `convert.py` with different parameters:

```import sys
import subprocess

def run_convert(config):
    cmd = ['python', 'convert.py'] + config.split()
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True)
        print(f'Running convert.py with parameter {config}:')
        print(output)
    except subprocess.CalledProcessError as e:
        print(f'Error running convert.py with parameter {config}:')
        print(e.output)

def run_multi_convert(mod):
    configs = (
        f"--torch-name-or-path {mod} --mlx-path mlx_models/{mod}_fp16",
        f"--torch-name-or-path {mod} --dtype float32 --mlx-path mlx_models/{mod}_fp32",
        f"--torch-name-or-path {mod} -q --q_bits 4 --mlx-path mlx_models/{mod}_quantized_4bits"
    )
    for config in configs:
        run_convert(config)

if __name__ == '__main__':
    # Check if models are provided, otherwise use default set
    if len(sys.argv) > 1:
        models = sys.argv[1:]
    else:
        models = ['tiny', 'tiny.en', 'small', 'small.en', 'medium', 'medium.en']  

    for mod in models:
        run_multi_convert(mod)
```

Save the above script as `wrapper.py`, and replace the default parameter list (`params`) with the 
desired set of strings to pass to `convert.py`. If no arguments are provided when calling 
`wrapper.py`, it will use the default parameters instead.

### Run

Transcribe audio with:

```python
import whisper

text = whisper.transcribe(speech_file)["text"]
```

Choose the model by setting `path_or_hf_repo`. For example:

```python
result = whisper.transcribe(speech_file, path_or_hf_repo="models/large")
```

This will load the model contained in `models/large`. The `path_or_hf_repo`
can also point to an MLX-style Whisper model on the Hugging Face Hub. In this
case, the model will be automatically downloaded.

The `transcribe` function also supports word-level timestamps. You can generate
these with:

```python
output = whisper.transcribe(speech_file, word_timestamps=True)
print(output["segments"][0]["words"])
```

To see more transcription options use:

```
>>> help(whisper.transcribe)
```

[^1]: Refer to the [arXiv paper](https://arxiv.org/abs/2212.04356), [blog post](https://openai.com/research/whisper), and [code](https://github.com/openai/whisper) for more details.
