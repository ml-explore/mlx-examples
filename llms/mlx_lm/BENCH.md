# MLX-LM Benchmark Tool

MLX-LM Benchmark Tool is a command-line utility for measuring and comparing the performance of MLX-format language models. It generates synthetic prompt tokens and captures various performance metrics, similar to llama.cpp's llama-bench tool.

## Features

- Measures multiple performance metrics:
  - Model load time (seconds)
  - Prompt token processing speed (TPS)
  - Generation token processing speed (TPS)
  - Total execution time (seconds)
  - Memory usage (GB)
- Supports testing combinations of multiple model configurations
- Customizable prompt token count and generation token count
- Multiple output formats (CSV, JSON, JSONL, Markdown)
- Configurable test repetitions for averaging performance

## Installation

Ensure you have MLX-LM installed:

```bash
pip install mlx-lm
```

## Usage

Basic usage:

```bash
mlx_lm.bench -m [MODEL_PATH] -p [PROMPT_TOKENS] -n [GEN_TOKENS] -r [REPETITIONS]
```

### Parameters

- `-m, --model`: Path to the MLX model(s) to benchmark, can specify multiple (comma-separated)
- `-p, --n-prompt`: Input Sequence Length (ISL), number of synthetic prompt tokens, can specify multiple (comma-separated)
- `-n, --n-gen`: Output Sequence Length (OSL), number of tokens to generate, can specify multiple (comma-separated)
- `-r, --repetitions`: Number of benchmark repetitions to average results over
- `-o, --output-format`: Output format for benchmark results (csv, json, jsonl, md)
- `-f, --output-filename`: Output filename (without extension)
- `--gen-args`: Additional keyword arguments for generate() function in key=value format

## Example

Benchmark two different Qwen models with different generation token counts:

```bash
mlx_lm.bench -m $HOME/Files/mlx/Qwen/Qwen2.5-3B-Instruct-Q4,$HOME/Files/mlx/Qwen/Qwen2.5-7B-Instruct-Q4 -p 1 -n 16,32 -r 2 -o md
```

Sample output:

| Model | Model Load Time (s) | Prompt Tokens | Prompt TPS | Response Tokens | Response TPS | Execution Time (s) | Memory Usage (GB) |
|---|---|---|---|---|---|---|---|
Qwen2.5-3B-Instruct-Q4 | 0.469 | 1 | 140.084 | 16 | 184.93 | 0.094 | 1.75 |
Qwen2.5-3B-Instruct-Q4 | 0.469 | 1 | 137.294 | 32 | 178.829 | 0.186 | 1.75 |
Qwen2.5-7B-Instruct-Q4 | 0.537 | 1 | 110.817 | 16 | 139.308 | 0.124 | 6.02 |
Qwen2.5-7B-Instruct-Q4 | 0.537 | 1 | 109.005 | 32 | 134.764 | 0.247 | 6.02 |

## Advanced Usage

Run more complex benchmarks:

```bash
# Test multiple models with various prompt and generation length combinations
mlx_lm.bench -m path/to/model1,path/to/model2 -p 1,8,64,128 -n 16,128,512 -r 3 -o json -f detailed_results

# Pass additional arguments to the generation function
mlx_lm.bench -m path/to/model -p 128 -n 128 -r 3 --gen-args kv_group_size=64
```

## Output Metrics

Benchmark results include the following metrics:

- **Model**: Path of the model being tested
- **Model Load Time (s)**: Time required to load the model (seconds)
- **Prompt Tokens**: Number of prompt tokens processed
- **Prompt TPS**: Prompt token processing speed (tokens per second)
- **Response Tokens**: Number of response tokens generated
- **Response TPS**: Response token generation speed (tokens per second)
- **Execution Time (s)**: Total execution time (seconds)
- **Memory Usage (GB)**: Peak memory usage (GB)