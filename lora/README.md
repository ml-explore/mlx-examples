# LoRA

This is an example of using MLX to fine-tune a Llama 7B[^llama] model with low
rank adaptation (LoRA)[^lora] for a target task. 

In this example we'll use the WikiSQL[^wikisql] dataset to train the LLM to
generate SQL queries from natural language. However, the example is intended to
be general should you wish to modify the task.

## Setup 

Install the dependencies:

```
pip install -r requirements.txt
```

Next, download and convert the model. If you do not have access to the model
weights you will need to [request
access](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform)
from Meta.

Convert the weights with:

```
python convert.py <path_to_torch_weights> mlx_llama_7B.npz
```

## Run

The main script is `lora.py`. To see a full list of options run

```
python lora.py --help
```

To fine-tune a model use:

```
python lora.py --model mlx_llama_7B.npz \
               --tokenizer tokenizer.model \
               --train \
               --iters 600 \
```

By default, the adapter weights are saved in `adapters.npz`. You can specify
the output location with `--adapter_file`.

To compute test set perplexity use

```
python lora.py --model mlx_llama_7B.npz \
               --tokenizer tokenizer.model \
               --data data \
               --test 
```

For generation use

```
python lora.py --model mlx_llama_7B.npz \
               --tokenizer tokenizer.model \
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

After training for 1000 iterations, the validation perplexity reduces to XX.

The model trains at around 475 tokens per second on an M2 Ultra.

[^lora]: Refer to the [arXiv paper](https://arxiv.org/abs/2106.09685) for more details on LoRA.
[^llama]: Refer to the [arXiv paper](https://arxiv.org/abs/2302.13971) and [blog post](https://ai.meta.com/blog/large-language-model-llama-meta-ai/) for more details.
[^wikisql]: Refer to the [GitHub repo](https://github.com/salesforce/WikiSQL/tree/master) for more information about WikiSQL.
