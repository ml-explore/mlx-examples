# LoRA

This is an example of using MLX to fine-tune either a Llama 7B[^llama] or a
Mistral 7B[^mistral] model with low rank adaptation (LoRA)[^lora] for a target
task. 

In this example we'll use the WikiSQL[^wikisql] dataset to train the LLM to
generate SQL queries from natural language. However, the example is intended to
be general should you wish to modify the task.

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
python convert.py <path_to_torch_model> <path_to_mlx_model>
```

## Run

#### Fine-tune

The main script is `lora.py`. To see a full list of options run

```
python lora.py --help
```

To fine-tune a model use:

```
python lora.py --model <path_to_model> \
               --train \
               --iters 600
```

Note, the model path should have the MLX weights, the tokenizer, and the
`params.json` configuration which will all be output by the `convert.py` script.

By default, the adapter weights are saved in `adapters.npz`. You can specify
the output location with `--adapter_file`.

#### Evaluate

To compute test set perplexity use

```
python lora.py --model <path_to_model> \
               --adapter_file <path_to_adapters.npz> \
               --test 
```

#### Generate

For generation use

```
python lora.py --model <path_to_model> \
               --adapter_file <path_to_adapters.npz> \
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

[^lora]: Refer to the [arXiv paper](https://arxiv.org/abs/2106.09685) for more details on LoRA.
[^llama]: Refer to the [arXiv paper](https://arxiv.org/abs/2302.13971) and [blog post](https://ai.meta.com/blog/large-language-model-llama-meta-ai/) for more details.
[^mistral]: Refer to the [blog post](https://mistral.ai/news/announcing-mistral-7b/) and [github repository](https://github.com/mistralai/mistral-src) for more details.
[^wikisql]: Refer to the [GitHub repo](https://github.com/salesforce/WikiSQL/tree/master) for more information about WikiSQL.
