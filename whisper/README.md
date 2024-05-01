# Whisper

Speech recognition with Whisper in MLX. Whisper is a set of open source speech
recognition models from OpenAI, ranging from 39 million to 1.5 billion
parameters.[^1]

### Setup

Install [`ffmpeg`](https://ffmpeg.org/):

```
# on macOS using Homebrew (https://brew.sh/)
brew install ffmpeg
```

Install the `mlx-whisper` package with:

```
pip install mlx-whisper
```

### Run

Transcribe audio with:

```python
import mlx_whisper

text = mlx_whisper.transcribe(speech_file)["text"]
```

The default model is "mlx-community/whisper-tiny". Choose the model by
setting `path_or_hf_repo`. For example:

```python
result = mlx_whisper.transcribe(speech_file, path_or_hf_repo="models/large")
```

This will load the model contained in `models/large`. The `path_or_hf_repo` can
also point to an MLX-style Whisper model on the Hugging Face Hub. In this case,
the model will be automatically downloaded. A [collection of pre-converted
Whisper
models](https://huggingface.co/collections/mlx-community/whisper-663256f9964fbb1177db93dc)
are in the Hugging Face MLX Community.

The `transcribe` function also supports word-level timestamps. You can generate
these with:

```python
output = mlx_whisper.transcribe(speech_file, word_timestamps=True)
print(output["segments"][0]["words"])
```

To see more transcription options use:

```
>>> help(mlx_whisper.transcribe)
```

### Converting models

> [!TIP]
> Skip the conversion step by using pre-converted checkpoints from the Hugging
> Face Hub. There are a few available in the [MLX
> Community](https://huggingface.co/mlx-community) organization.

To convert a model, first clone the MLX Examples repo:

```
git clone https://github.com/ml-explore/mlx-examples.git
```

Then run `convert.py` from `mlx-examples/whisper`. For example, to convert the
`tiny` model use:

```
python convert.py --torch-name-or-path tiny --mlx-path mlx_models/tiny
```

Note you can also convert a local PyTorch checkpoint which is in the original
OpenAI format.

To generate a 4-bit quantized model, use `-q`. For a full list of options:

```
python convert.py --help
```

By default, the conversion script will make the directory `mlx_models`
and save the converted `weights.npz` and `config.json` there. 

Each time it is run, `convert.py` will overwrite any model in the provided
path. To save different models, make sure to set `--mlx-path` to a unique
directory for each converted model. For example:

```bash
model="tiny"
python convert.py --torch-name-or-path ${model} --mlx-path mlx_models/${model}_fp16
python convert.py --torch-name-or-path ${model} --dtype float32 --mlx-path mlx_models/${model}_fp32
python convert.py --torch-name-or-path ${model} -q --q_bits 4 --mlx-path mlx_models/${model}_quantized_4bits
```

[^1]: Refer to the [arXiv paper](https://arxiv.org/abs/2212.04356), [blog post](https://openai.com/research/whisper), and [code](https://github.com/openai/whisper) for more details.
