# MusicGen

An example of Meta's MusicGen model in MLX.[^1] MusicGen is used to generate
music from text descriptions.

### Setup

Install the requirements:

```
pip install -r requirements.txt
```

Optionally install FFmpeg and SciPy for loading and saving audio files,
respectively.

Install [FFmpeg](https://ffmpeg.org/):

```
# on macOS using Homebrew (https://brew.sh/)
brew install ffmpeg
```

Install SciPy:

```
pip install scipy
```

### Example

An example using the model:

```python
import mlx.core as mx
from music_gen import MusicGen
from utils import save_audio

# Load the 48 KHz model and preprocessor.
model, processor = MusicGen.from_pretrained("facebook/musicgen-medium")

audio = model.generate("happy rock")

# Save the audio as a wave file
save_audio("out.wav", audio, model.sampling_rate)
```

[^1]: Refer to the [arXiv paper](https://arxiv.org/abs/2306.05284) and
  [code](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md) for more details.
