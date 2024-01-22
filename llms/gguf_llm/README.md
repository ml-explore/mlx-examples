# LLMs in MLX with GGUF

An example generating text using GGUF format models in MLX.[^1]

> [!NOTE]
> MLX is able to read most quantization formats from GGUF directly. However,
> only a few quantizations are supported directly: `Q4_0`, `Q4_1`, and `Q8_0`.
> Unsupported quantizations will be cast to `float16`.

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Run

Run with:

```bash
python generate.py \
  --repo <hugging_face_repo> \
  --gguf <file.gguf> \
  --prompt "Write a quicksort in Python"
```

For example, to generate text with Mistral 7B use:

```bash
python generate.py \
  --repo TheBloke/Mistral-7B-v0.1-GGUF \
  --gguf mistral-7b-v0.1.Q8_0.gguf \
  --prompt "Write a quicksort in Python"
```

Run `python generate.py --help` for more options.

Models that have been tested and work include:

- [TheBloke/Mistral-7B-v0.1-GGUF](https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF),
  for quantized models use:
  - `mistral-7b-v0.1.Q8_0.gguf`
  - `mistral-7b-v0.1.Q4_0.gguf`

- [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF),
  for quantized models use:
  - `tinyllama-1.1b-chat-v1.0.Q8_0.gguf`
  - `tinyllama-1.1b-chat-v1.0.Q4_0.gguf` 

[^1]: For more information on GGUF see [the documentation](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md).
