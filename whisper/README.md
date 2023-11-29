# whisper

Whisper in MLX.

First install the dependencies:

(TODO, MLX install link / command / add to requirements.txt)

```
pip install -r requirements.txt
```

Install [`ffmpeg`](https://ffmpeg.org/):

```bash
# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg
```

Then transcribe audio with:

```
import whisper

text = whisper.transcribe(speech_file)["text"]
```

