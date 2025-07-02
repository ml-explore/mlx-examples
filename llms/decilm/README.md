# DeciLM / Nemotron-NAS Support for MLX

This module provides native MLX support for DeciLM architecture models, including NVIDIA's Nemotron series. DeciLM uses Neural Architecture Search (NAS) to create highly optimized transformer variants that achieve superior performance through architectural innovations.

## Architecture Features

DeciLM uses Neural Architecture Search (NAS) optimization with:

1. **Dummy Layers**: Layers where attention or FFN components are completely removed
2. **FFN Fusion**: Multiple sequential FFN layers fused into wider parallel layers  
3. **Variable Grouped Query Attention (VGQA)**: Different number of KV heads per layer (1-8)

## Supported Models

- nvidia/Llama-3_1-Nemotron-Ultra-253B-v1
- nvidia/Llama-3_1-Nemotron-51B-Instruct
- Other DeciLM-based models

## Usage

### Converting Models

```bash
python convert.py \
    --hf-path nvidia/Llama-3_1-Nemotron-Ultra-253B-v1 \
    --mlx-path ./nemotron-253b-mlx \
    --quantize --q-bits 5
```

### Loading and Generation

```python
from mlx_lm import load, generate
from decilm import Model, DeciLMArgs

# Load pre-converted model
model, tokenizer = load("./nemotron-253b-mlx")

# Generate text
response = generate(
    model, 
    tokenizer, 
    prompt="Explain quantum computing in simple terms",
    max_tokens=500,
    temperature=0.7,
    verbose=True
)
print(response)
```

### Command Line Usage

```bash
# Using mlx_lm CLI
mlx_lm.generate \
    --model ./nemotron-253b-mlx \
    --prompt "Your prompt here" \
    --max-tokens 1000 \
    --temperature 0.8

# Start API server
mlx_lm.server \
    --model ./nemotron-253b-mlx \
    --host 0.0.0.0 \
    --port 8080
```

## Implementation Details

The implementation handles:
- Block configurations with variable architectures
- Dummy layer passthrough (no computation)
- FFN fusion for improved efficiency
- Per-layer attention head configuration

## Performance

Tested on Mac Studio M3 Ultra (512GB RAM):
- Nemotron-253B Q5: ~3.86 tokens/sec generation
- Memory usage: ~175GB peak

## LM Studio Compatibility

⚠️ **Note**: DeciLM models are currently **NOT compatible with LM Studio** due to the NAS architecture with dummy layers. LM Studio expects standard transformer layers and encounters "NoneType object has no attribute 'shape'" errors with dummy components.

**Use `mlx_lm` CLI tools instead:**
```bash
# Generate text
uv run mlx_lm.generate \
  --model /path/to/nemotron-mlx \
  --prompt "Your prompt here" \
  --max-tokens 1000

# Start server
uv run mlx_lm.server \
  --model /path/to/nemotron-mlx \
  --host 0.0.0.0 \
  --port 8080
```

### Tokenizer Issues

If you encounter tokenizer issues, check the `USE-IF-MODEL-FAILED-TO-GENERATE` subfolder in the model directory for patched tokenizer configs and chat templates.

## Requirements

- **MLX**: >= 0.26.1 
- **Python**: 3.11 - 3.12 (tested with CPython 3.12.11 via `uv`)
- **Memory**: Sufficient RAM for model size (e.g., ~175GB for Nemotron-253B)
- **mlx-lm**: Latest version for model inference

## Production Deployment

For production-grade API deployment, consider using [**lbrxServer**](https://github.com/LibraxisAI/lbrxServer):
- Robust API endpoints for various LLM architectures
- Native support for DeciLM/Nemotron models
- Built-in load balancing and request queuing
- Compatible with OpenAI API format

## Model Availability

Pre-converted DeciLM models for MLX:
- [LibraxisAI/Llama-3_1-Nemotron-Ultra-253B-v1-mlx-q5](https://huggingface.co/LibraxisAI/Llama-3_1-Nemotron-Ultra-253B-v1-mlx-q5) - 253B Q5 quantized

## Testing

Run the test suite:
```bash
cd tests
python -m pytest test_decilm.py -v
```

For integration testing with a real model:
```bash
python test_generation.py --model-path /path/to/decilm-model
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This module follows the same license as mlx-examples. Model weights are subject to their original licenses.