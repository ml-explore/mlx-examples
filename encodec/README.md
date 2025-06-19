# EnCodec

An example of Meta's EnCodec model in MLX.[^1] EnCodec is used to compress and
generate audio.

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
from encodec import EncodecModel
from utils import load_audio, save_audio

# Load the 48 KHz model and preprocessor.
model, processor = EncodecModel.from_pretrained("mlx-community/encodec-48khz-float32")

# Load an audio file
audio = load_audio("path/to/audio", model.sampling_rate, model.channels)

# Preprocess the audio (this can also be a list of arrays for batched
# processing).
feats, mask = processor(audio)

# Encode at the given bandwidth. A lower bandwidth results in more
# compression but lower reconstruction quality.
@mx.compile
def encode(feats, mask):
    return model.encode(feats, mask, bandwidth=3)

# Decode to reconstruct the audio
@mx.compile
def decode(codes, scales, mask):
    return model.decode(codes, scales, mask)


codes, scales = encode(feats, mask)
reconstructed = decode(codes, scales, mask)

# Trim any padding:
reconstructed = reconstructed[0, : len(audio)]

# Save the audio as a wave file
save_audio("reconstructed.wav", reconstructed, model.sampling_rate)
```

The 24 KHz, 32 KHz, and 48 KHz MLX formatted models are available in the
[Hugging Face MLX Community](https://huggingface.co/collections/mlx-community/encodec-66e62334038300b07a43b164)
in several data types.

### Optional

To convert models, use the `convert.py` script. To see the options, run:

```bash
python convert.py -h
```

[^1]: Refer to the [arXiv paper](https://arxiv.org/abs/2210.13438) and
  [code](https://github.com/facebookresearch/encodec) for more details.
