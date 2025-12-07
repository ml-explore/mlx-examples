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

#### CLI

At its simplest:

```sh
mlx_whisper audio_file.mp3
```

This will make a text file `audio_file.txt` with the results.

Use `-f` to specify the output format and `--model` to specify the model. There
are many other supported command line options. To see them all, run
`mlx_whisper -h`.

You can also pipe the audio content of other programs via stdin:

```sh
some-process | mlx_whisper -
```

The default output file name will be `content.*`. You can specify the name with
the `--output-name` flag.

#### API

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

### Beam Search Decoding

By default, mlx-whisper uses greedy decoding. Enable beam search for potentially
more accurate transcriptions at the cost of speed:

```bash
# Enable beam search with beam size 5
mlx_whisper audio.mp3 --beam-size 5

# Adjust patience for earlier/later stopping (default: 1.0)
mlx_whisper audio.mp3 --beam-size 5 --patience 1.5
```

In Python:

```python
result = mlx_whisper.transcribe(
    "audio.mp3",
    beam_size=5,
    patience=1.0
)
```

The `patience` parameter controls early stopping: decoding stops when
`round(beam_size * patience)` finished sequences have been collected.
Higher patience values explore more candidates before stopping.

### Voice Activity Detection (VAD)

Enable Silero VAD to filter silent audio regions before transcription. This can
significantly speed up transcription for audio with long silent periods:

```bash
# Enable VAD
mlx_whisper audio.mp3 --vad-filter

# Customize VAD settings
mlx_whisper audio.mp3 --vad-filter --vad-threshold 0.6 --vad-min-silence-ms 1000
```

In Python:

```python
from mlx_whisper import transcribe
from mlx_whisper.vad import VadOptions

result = transcribe("audio.mp3", vad_filter=True)

# With custom options
vad_opts = VadOptions(threshold=0.6, min_silence_duration_ms=1000)
result = transcribe("audio.mp3", vad_filter=True, vad_options=vad_opts)
```

**Requirements**: `pip install torch`

### Speaker Diarization

Identify who is speaking when with pyannote.audio. Diarization adds speaker
labels to transcription segments:

```bash
# Enable diarization (requires HuggingFace token)
export HF_TOKEN=your_token
mlx_whisper audio.mp3 --diarize --word-timestamps

# Specify speaker count
mlx_whisper audio.mp3 --diarize --min-speakers 2 --max-speakers 4

# Output diarization in RTTM format
mlx_whisper audio.mp3 --diarize -f rttm
```

In Python:

```python
from mlx_whisper import transcribe_with_diarization

result = transcribe_with_diarization(
    "audio.mp3",
    hf_token="your_token",
    word_timestamps=True
)

# Access speaker info
for segment in result["segments"]:
    speaker = segment.get("speaker", "Unknown")
    print(f"{speaker}: {segment['text']}")

# List of speakers
print(result["speakers"])  # ['SPEAKER_00', 'SPEAKER_01', ...]
```

**Requirements**:
- `pip install pyannote.audio pandas`
- Accept model terms at https://huggingface.co/pyannote/speaker-diarization-3.1
- Set `HF_TOKEN` environment variable or pass `--hf-token`

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
