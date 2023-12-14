# Whisper

Speech recognition with Whisper in MLX. Whisper is a set of open source speech
recognition models from OpenAI, ranging from 39 million to 1.5 billion
parameters[^1].

### Setup

First, install the dependencies.

```
pip install -r requirements.txt
```

Install [`ffmpeg`](https://ffmpeg.org/):

```
# on macOS using Homebrew (https://brew.sh/)
brew install ffmpeg
```

### Run

Transcribe audio with:

```
import whisper

text = whisper.transcribe(speech_file)["text"]
```

[^1]: Refer to the [arXiv paper](https://arxiv.org/abs/2212.04356), [blog post](https://openai.com/research/whisper), and [code](https://github.com/openai/whisper) for more details.
