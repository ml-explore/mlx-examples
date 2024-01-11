# LLMs in MLX with GGUF

An example of running GGUF models using MLX without having to pre-convert the
weights.[^1]

> [!NOTE]
> MLX is able to read most quantization formats from GGUF directly. However,
> currently all quantized formats will be cast to `float16`. So the following
> models will be run in 16-bit precision.

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Run

For example, to generate text with Mistral 7B use:

```bash
python generate.py \
  --repo TheBloke/Mistral-7B-v0.1-GGUF \
  --gguf mistral-7b-v0.1.Q6_K.gguf \
  --prompt "hello" \
```

To generate text with TinyLlama use:

```bash
python generate.py \
  --repo TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  --gguf tinyllama-1.1b-chat-v1.0.Q6_K.gguf \
  --prompt "hello" \
```

Run `python generate.py --help` for more options.

[^1]: For more information on GGUF see [the documentation](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md).
