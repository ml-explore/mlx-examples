# Fine-Tuning with LoRA or QLoRA

You can use use the `mlx-lm` package to fine-tune an LLM with low rank
adaptation (LoRA) for a target task.[^lora] The example also supports quantized
LoRA (QLoRA).[^qlora] LoRA fine-tuning works with the following model families:

- Mistral
- Llama
- Phi2
- Mixtral
- Qwen2
- Gemma
- OLMo
- MiniCPM

## Contents

- [Run](#Run)
  - [Fine-tune](#Fine-tune)
  - [Evaluate](#Evaluate)
  - [Generate](#Generate)
- [Fuse](#Fuse)
- [Data](#Data)
- [Memory Issues](#Memory-Issues)

## Run

The main command is `mlx_lm.lora`. To see a full list of command-line options run:

```shell
mlx_lm.lora --help
```

Note, in the following the `--model` argument can be any compatible Hugging
Face repo or a local path to a converted model.

You can also specify a YAML config with `-c`/`--config`. For more on the format see the
[example YAML](examples/lora_config.yaml). For example:

```shell
mlx_lm.lora --config /path/to/config.yaml
```

If command-line flags are also used, they will override the corresponding
values in the config.

### Fine-tune

To fine-tune a model use:

```shell
mlx_lm.lora \
    --model <path_to_model> \
    --train \
    --data <path_to_data> \
    --iters 600
```

The `--data` argument must specify a path to a `train.jsonl`, `valid.jsonl`
when using `--train` and a path to a `test.jsonl` when using `--test`. For more
details on the data format see the section on [Data](#Data).

For example, to fine-tune a Mistral 7B you can use `--model
mistralai/Mistral-7B-v0.1`.

If `--model` points to a quantized model, then the training will use QLoRA,
otherwise it will use regular LoRA.

By default, the adapter config and weights are saved in `adapters/`. You can
specify the output location with `--adapter-path`.

You can resume fine-tuning with an existing adapter with
`--resume-adapter-file <path_to_adapters.safetensors>`.

### Evaluate

To compute test set perplexity use:

```shell
mlx_lm.lora \
    --model <path_to_model> \
    --adapter-path <path_to_adapters> \
    --data <path_to_data> \
    --test
```

### Generate

For generation use `mlx_lm.generate`:

```shell
mlx_lm.generate \
    --model <path_to_model> \
    --adapter-path <path_to_adapters> \
    --prompt "<your_model_prompt>"
```

## Fuse

You can generate a model fused with the low-rank adapters using the
`mlx_lm.fuse` command. This command also allows you to optionally:

- Upload the fused model to the Hugging Face Hub.
- Export the fused model to GGUF. Note GGUF support is limited to Mistral,
  Mixtral, and Llama style models in fp16 precision.

To see supported options run:

```shell
mlx_lm.fuse --help
```

To generate the fused model run:

```shell
mlx_lm.fuse --model <path_to_model>
```

This will by default load the adapters from `adapters/`, and save the fused
model in the path `lora_fused_model/`. All of these are configurable.

To upload a fused model, supply the `--upload-repo` and `--hf-path` arguments
to `mlx_lm.fuse`. The latter is the repo name of the original model, which is
useful for the sake of attribution and model versioning.

For example, to fuse and upload a model derived from Mistral-7B-v0.1, run:

```shell
mlx_lm.fuse \
    --model mistralai/Mistral-7B-v0.1 \
    --upload-repo mlx-community/my-4bit-lora-mistral \
    --hf-path mistralai/Mistral-7B-v0.1
```

To export a fused model to GGUF, run:

```shell
mlx_lm.fuse \
    --model mistralai/Mistral-7B-v0.1 \
    --export-gguf
```

This will save the GGUF model in `lora_fused_model/ggml-model-f16.gguf`. You
can specify the file name with `--gguf-path`.

## Data

The LoRA command expects you to provide a dataset with `--data`. The MLX
Examples GitHub repo has an [example of the WikiSQL
data](https://github.com/ml-explore/mlx-examples/tree/main/lora/data) in the
correct format.

For fine-tuning (`--train`), the data loader expects a `train.jsonl` and a
`valid.jsonl` to be in the data directory. For evaluation (`--test`), the data
loader expects a `test.jsonl` in the data directory.

Currently, `*.jsonl` files support three data formats: `chat`,
`completions`, and `text`. Here are three examples of these formats:

`chat`:

```jsonl
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello."
    },
    {
      "role": "assistant",
      "content": "How can I assistant you today."
    }
  ]
}
```

`completions`:

```jsonl
{
  "prompt": "What is the capital of France?",
  "completion": "Paris."
}
```

`text`:

```jsonl
{
  "text": "This is an example for the model."
}
```

Note, the format is automatically determined by the dataset. Note also, keys in
each line not expected by the loader will be ignored.

For the `chat` and `completions` formats, Hugging Face [chat
templates](https://huggingface.co/blog/chat-templates) are used. This applies
the model's chat template by default. If the model does not have a chat
template, then Hugging Face will use a default. For example, the final text in
the `chat` example above with Hugging Face's default template becomes:

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello.<|im_end|>
<|im_start|>assistant
How can I assistant you today.<|im_end|>
```

If you are unsure of the format to use, the `chat` or `completions` are good to
start with. For custom requirements on the format of the dataset, use the
`text` format to assemble the content yourself.

## Memory Issues

Fine-tuning a large model with LoRA requires a machine with a decent amount
of memory. Here are some tips to reduce memory use should you need to do so:

1. Try quantization (QLoRA). You can use QLoRA by generating a quantized model
   with `convert.py` and the `-q` flag. See the [Setup](#setup) section for
   more details.

2. Try using a smaller batch size with `--batch-size`. The default is `4` so
   setting this to `2` or `1` will reduce memory consumption. This may slow
   things down a little, but will also reduce the memory use.

3. Reduce the number of layers to fine-tune with `--lora-layers`. The default
   is `16`, so you can try `8` or `4`. This reduces the amount of memory
   needed for back propagation. It may also reduce the quality of the
   fine-tuned model if you are fine-tuning with a lot of data.

4. Longer examples require more memory. If it makes sense for your data, one thing
   you can do is break your examples into smaller
   sequences when making the `{train, valid, test}.jsonl` files.

5. Gradient checkpointing lets you trade-off memory use (less) for computation
   (more) by recomputing instead of storing intermediate values needed by the
   backward pass. You can use gradient checkpointing by passing the
   `--grad-checkpoint` flag. Gradient checkpointing will be more helpful for
   larger batch sizes or sequence lengths with smaller or quantized models.

For example, for a machine with 32 GB the following should run reasonably fast:

```
python lora.py \
    --model mistralai/Mistral-7B-v0.1 \
    --train \
    --batch-size 1 \
    --lora-layers 4 \
    --data wikisql
```

The above command on an M1 Max with 32 GB runs at about 250
tokens-per-second, using the MLX Example
[`wikisql`](https://github.com/ml-explore/mlx-examples/tree/main/lora/data)
data set.

[^lora]: Refer to the [arXiv paper](https://arxiv.org/abs/2106.09685) for more details on LoRA.
[^qlora]: Refer to the paper [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
