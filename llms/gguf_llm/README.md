# Llama

An example of running GGUF models using MLX without having to convert the weights.

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Run

```bash
python llama.py --repo TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --gguf tinyllama-1.1b-chat-v1.0.Q6_K.gguf --prompt "hello"
```

Run `python llama.py --help` for more details.
