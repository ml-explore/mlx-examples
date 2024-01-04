# Llama

An example of running GGUF models using MLX without having to convert the weights.

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

Next, download the model.

```bash
mkdir -p models/tiny_llama
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q6_K.gguf \
  -O models/tiny_llama/model.gguf
# TODO: gguf should be sufficient
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/config.json \
  -O models/tiny_llama/config.json
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.json \
  -O models/tiny_llama/tokenizer.json
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.model \
  -O models/tiny_llama/tokenizer.model
```

### Run

```bash
python llama.py --prompt "hello"
```

Run `python llama.py --help` for more details.
