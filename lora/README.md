# Fine-Tuning with LoRA or QLoRA

This is an example of using MLX to fine-tune an LLM with low rank adaptation
(LoRA) for a target task.[^lora] The example also supports quantized LoRA
(QLoRA).[^qlora] The example works with Llama and Mistral style models
available on Hugging Face.

> [!TIP]
> For a more fully featured LLM package, checkout [MLX
> LM](https://github.com/ml-explore/mlx-lm).

In this example we'll use the WikiSQL[^wikisql] dataset to train the LLM to
generate SQL queries from natural language. However, the example is intended to
be general should you wish to use a custom dataset.

## Contents

* [Setup](#Setup)
  * [Convert](#convert)
* [Run](#Run)
  * [Fine-tune](#Fine-tune)
  * [Evaluate](#Evaluate)
  * [Generate](#Generate)
* [Results](#Results)
* [Fuse and Upload](#Fuse-and-Upload)
* [Custom Data](#Custom-Data)
* [Memory Issues](#Memory-Issues)


## Setup 

Install the dependencies:

```
pip install -r requirements.txt
```

### Convert

This step is optional if you want to quantize (for QLoRA) or change the default
data type of a pre-existing model.

You convert models using the `convert.py` script. This script takes a Hugging
Face repo as input and outputs a model directory (which you can optionally also
upload to Hugging Face).

To make a 4-bit quantized model, run:

```
python convert.py --hf-path <hf_repo> -q
```

For example, the following will make a 4-bit quantized Mistral 7B and by default
store it in `mlx_model`:

```
python convert.py --hf-path mistralai/Mistral-7B-v0.1 -q
```

For more options run:

```
python convert.py --help
```

You can upload new models to the [Hugging Face MLX
Community](https://huggingface.co/mlx-community) by specifying `--upload-name`
to `convert.py`.

## Run

The main script is `lora.py`. To see a full list of options run:

```
python lora.py --help
```

Note, in the following the `--model` argument can be any compatible Hugging
Face repo or a local path to a converted mdoel. 

### Fine-tune

To fine-tune a model use:

```
python lora.py --model <path_to_model> \
               --train \
               --iters 600
```

If `--model` points to a quantized model, then the training will use QLoRA,
otherwise it will use regular LoRA.

By default, the adapter weights are saved in `adapters.npz`. You can specify
the output location with `--adapter-file`.

You can resume fine-tuning with an existing adapter with `--resume-adapter-file
<path_to_adapters.npz>`. 

### Evaluate

To compute test set perplexity use:

```
python lora.py --model <path_to_model> \
               --adapter-file <path_to_adapters.npz> \
               --test
```

### Generate

For generation use:

```
python lora.py --model <path_to_model> \
               --adapter-file <path_to_adapters.npz> \
               --max-tokens 50 \
               --prompt "table: 1-10015132-16
columns: Player, No., Nationality, Position, Years in Toronto, School/Club Team
Q: What is terrence ross' nationality
A: "
```

## Results

The initial validation loss for Llama 7B on the WikiSQL is 2.66 and the final
validation loss after 1000 iterations is 1.23. The table below shows the
training and validation loss at a few points over the course of training.

| Iteration | Train Loss | Validation Loss |
| --------- | ---------- | --------------- |
| 1         |    N/A     |      2.659      |
| 200       |    1.264   |      1.405      |
| 400       |    1.201   |      1.303      |
| 600       |    1.123   |      1.274      |
| 800       |    1.017   |      1.255      |
| 1000      |    1.070   |      1.230      |

The model trains at around 475 tokens per second on an M2 Ultra.

## Fuse and Upload

You can generate a fused model with the low-rank adapters included using the
`fuse.py` script. This script also optionally allows you to upload the fused
model to the [Hugging Face MLX
Community](https://huggingface.co/mlx-community).

To generate the fused model run:

```
python fuse.py
```

This will by default load the base model from `mlx_model/`, the adapters from
`adapters.npz`,  and save the fused model in the path `lora_fused_model/`. All
of these are configurable. You can see the list of options with:

```
python fuse.py --help
```

To upload a fused model, supply the `--upload-name` and `--hf-path` arguments
to `fuse.py`. The latter is the repo name of the original model, which is
useful for the sake of attribution and model versioning.

For example, to fuse and upload a model derived from Mistral-7B-v0.1, run: 

```
python fuse.py --upload-name My-4-bit-model --hf-path mistralai/Mistral-7B-v0.1
```

## Custom Data

You can make your own dataset for fine-tuning with LoRA. You can specify the
dataset with `--data=<my_data_directory>`. Check the subdirectory `data/` to
see the expected format.

For fine-tuning (`--train`), the data loader expects a `train.jsonl` and a
`valid.jsonl` to be in the data directory. For evaluation (`--test`), the data
loader expects a `test.jsonl` in the data directory. Each line in the `*.jsonl`
file should look like:

```
{"text": "This is an example for the model."}
```

Note other keys will be ignored by the loader.

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
   --lora-layers 4
```

The above command on an M1 Max with 32 GB runs at about 250 tokens-per-second.


[^lora]: Refer to the [arXiv paper](https://arxiv.org/abs/2106.09685) for more details on LoRA.
[^qlora]: Refer to the paper [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
[^wikisql]: Refer to the [GitHub repo](https://github.com/salesforce/WikiSQL/tree/master) for more information about WikiSQL.
