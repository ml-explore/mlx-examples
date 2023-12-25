# LoRA

This is an example of using MLX to fine-tune either a Llama 7B[^llama] or a
Mistral 7B[^mistral] model with low rank adaptation (LoRA)[^lora] for a target
task. 

In this example we'll use the WikiSQL[^wikisql] dataset to train the LLM to
generate SQL queries from natural language. However, the example is intended to
be general should you wish to use a custom dataset.

## Contents

* [Setup](#Setup)
* [Run](#Run)
  * [Fine-tune](#Fine-tune)
  * [Evaluate](#Evaluate)
  * [Generate](#Generate)
* [Results](#Results)
* [Custom Data](#Custom-Data)
* [Memory Issues](#Memory-Issues)


## Setup 

Install the dependencies:

```
pip install -r requirements.txt
```

Next, download and convert the model. The Mistral weights can be downloaded with:

```
curl -O https://files.mistral-7b-v0-1.mistral.ai/mistral-7B-v0.1.tar
tar -xf mistral-7B-v0.1.tar
```

If you do not have access to the Llama weights you will need to [request
access](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)
from Meta.

Convert the model with:

```
python convert.py \
    --torch-model <path_to_torch_model> \
    --mlx-model <path_to_mlx_model>
```

## Run

The main script is `lora.py`. To see a full list of options run

```
python lora.py --help
```

### Fine-tune

To fine-tune a model use:

```
python lora.py --model <path_to_model> \
               --train \
               --iters 600
```

Note, the model path should have the MLX weights, the tokenizer, and the
`params.json` configuration which will all be output by the `convert.py` script.

By default, the adapter weights are saved in `adapters.npz`. You can specify
the output location with `--adapter-file`.

You can resume fine-tuning with an existing adapter with `--resume-adapter-file
<path_to_adapters.npz>`. 

### Evaluate

To compute test set perplexity use

```
python lora.py --model <path_to_model> \
               --adapter-file <path_to_adapters.npz> \
               --test 
```

### Generate

For generation use

```
python lora.py --model <path_to_model> \
               --adapter-file <path_to_adapters.npz> \
               --num-tokens 50 \
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

1. Try using a smaller batch size with `--batch-size`. The default is `4` so
   setting this to `2` or `1` will reduce memory consumption. This may slow
   things down a little, but will also reduce the memory use.

2. Reduce the number of layers to fine-tune with `--lora-layers`. The default
   is `16`, so you can try `8` or `4`. This reduces the amount of memory
   needed for back propagation. It may also reduce the quality of the
   fine-tuned model if you are fine-tuning with a lot of data.

3. Longer examples require more memory. If it makes sense for your data, one thing
   you can do is break your examples into smaller
   sequences when making the `{train, valid, test}.jsonl` files.

For example, for a machine with 32 GB the following should run reasonably fast:

```
python lora.py \
   --model <path_to_model> \
   --train \
   --batch-size 1 \
   --lora-layers 4
```

The above command on an M1 Max with 32 GB runs at about 250 tokens-per-second.


[^lora]: Refer to the [arXiv paper](https://arxiv.org/abs/2106.09685) for more details on LoRA.
[^llama]: Refer to the [arXiv paper](https://arxiv.org/abs/2302.13971) and [blog post](https://ai.meta.com/blog/large-language-model-llama-meta-ai/) for more details.
[^mistral]: Refer to the [blog post](https://mistral.ai/news/announcing-mistral-7b/) and [github repository](https://github.com/mistralai/mistral-src) for more details.
[^wikisql]: Refer to the [GitHub repo](https://github.com/salesforce/WikiSQL/tree/master) for more information about WikiSQL.
