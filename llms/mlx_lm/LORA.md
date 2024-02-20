# Fine-Tuning with LoRA or QLoRA

You can use use the `mlx-lm` package to fine-tune an LLM with low rank
adaptation (LoRA) for a target task.[^lora] The example also supports quantized
LoRA (QLoRA).[^qlora] LoRA fine-tuning works with the following model families:

- Mistral
- Llama
- Phi2
- Mixtral
- Qwen2
- OLMo

## Contents

* [Run](#Run)
  * [Fine-tune](#Fine-tune)
  * [Evaluate](#Evaluate)
  * [Generate](#Generate)
* [Fuse and Upload](#Fuse-and-Upload)
* [Data](#Data)
* [Memory Issues](#Memory-Issues)

## Run

The main command is `mlx_lm.lora`. To see a full list of options run:

```shell
python -m mlx_lm.lora --help
```

Note, in the following the `--model` argument can be any compatible Hugging
Face repo or a local path to a converted model. 

### Fine-tune

To fine-tune a model use:

```shell
python -m mlx_lm.lora \
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

By default, the adapter weights are saved in `adapters.npz`. You can specify
the output location with `--adapter-file`.

You can resume fine-tuning with an existing adapter with
`--resume-adapter-file <path_to_adapters.npz>`. 

### Evaluate

To compute test set perplexity use:

```shell
python -m mlx_lm.lora \
    --model <path_to_model> \
    --adapter-file <path_to_adapters.npz> \
    --data <path_to_data> \
    --test
```

## Fuse and Upload

You can generate a model fused with the low-rank adapters using the
`mlx_lm.fuse` command. This command also allows you to upload the fused model
to the Hugging Face Hub.

To see supported options run:

```shell
python -m mlx_lm.fuse --help
```

To generate the fused model run:

```shell
python -m mlx_lm.fuse --model <path_to_model>
```

This will by default load the adapters from `adapters.npz`, and save the fused
model in the path `lora_fused_model/`. All of these are configurable.

To upload a fused model, supply the `--upload-repo` and `--hf-path` arguments
to `mlx_lm.fuse`. The latter is the repo name of the original model, which is
useful for the sake of attribution and model versioning.

For example, to fuse and upload a model derived from Mistral-7B-v0.1, run: 

```shell
python -m mlx_lm.fuse \
    --model mistralai/Mistral-7B-v0.1 \
    --upload-repo mlx-community/my-4bit-lora-mistral \
    --hf-path mistralai/Mistral-7B-v0.1
```

## Data

The LoRA command expects you to provide a dataset with `--data`.  The MLX
Examples GitHub repo has an [example of the WikiSQL
data](https://github.com/ml-explore/mlx-examples/tree/main/lora/data) in the
correct format.

For fine-tuning (`--train`), the data loader expects a `train.jsonl` and a
`valid.jsonl` to be in the data directory. For evaluation (`--test`), the data
loader expects a `test.jsonl` in the data directory. Each line in the `*.jsonl`
file should look like:

```
{"text": "This is an example for the model."}
```

Note, other keys will be ignored by the loader.

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
