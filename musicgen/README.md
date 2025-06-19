# MusicGen

An example of Meta's MusicGen model in MLX.[^1] MusicGen is used to generate
music from text descriptions.

### Setup

Install the requirements:

```
pip install -r requirements.txt
```

### Example

An example using the model:

```python
from musicgen import MusicGen
from utils import save_audio

model = MusicGen.from_pretrained("facebook/musicgen-medium")

audio = model.generate("happy rock")

save_audio("out.wav", audio, model.sampling_rate)
```

[^1]: Refer to the [arXiv paper](https://arxiv.org/abs/2306.05284) and
  [code](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md) for more details.
