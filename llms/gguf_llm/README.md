# LLMs in MLX with GGUF

An example generating text using GGUF format models in MLX.[^1]

> [!NOTE]
> MLX is able to read most quantization formats from GGUF directly. However,
> currently all quantized formats will be cast to `float16`. In the following
> models will be run in 16-bit precision.

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
  --prompt "hello"
```

For example, to generate text with Mistral 7B use:

```bash
python generate.py \
  --repo TheBloke/Mistral-7B-v0.1-GGUF \
  --gguf mistral-7b-v0.1.Q6_K.gguf \
  --prompt "hello"
```

Run `python generate.py --help` for more options.

[^1]: For more information on GGUF see [the documentation](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md).
