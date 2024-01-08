# Llama

An example of running GGUF models using MLX without having to convert the weights.

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

Next, download the model.

```bash
mkdir -p models/
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q6_K.gguf \
  -P models/
```

### Run

```bash
python llama.py --model models/tinyllama-1.1b-chat-v1.0.Q6_K.gguf --prompt "hello"
```

Run `python llama.py --help` for more details.
