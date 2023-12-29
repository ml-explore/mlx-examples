# Whisper

Speech recognition with Whisper in MLX. Whisper is a set of open source speech
recognition models from OpenAI, ranging from 39 million to 1.5 billion
parameters[^1].

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

Next, download the Whisper PyTorch checkpoint and convert the weights to the MLX format. For example, to convert the `tiny` model use:

```
python convert.py --torch-name-or-path tiny --mlx-path mlx_models/tiny
```

Note you can also convert a local PyTorch checkpoint which is in the original OpenAI format.

To generate a 4-bit quantized model, use `-q`. For a full list of options:

```
python convert.py --help
```

By default, the conversion script will make the directory `mlx_models/tiny` and save
the converted `weights.npz` and `config.json` there.

> [!TIP]
> Alternatively, you can also download a few converted checkpoints from the
> [MLX Community](https://huggingface.co/mlx-community) organization on Hugging
> Face and skip the conversion step.

### Run

Transcribe audio with:

```
import whisper

text = whisper.transcribe(speech_file)["text"]
```

[^1]: Refer to the [arXiv paper](https://arxiv.org/abs/2212.04356), [blog post](https://openai.com/research/whisper), and [code](https://github.com/openai/whisper) for more details.
