# T5

The T5 models are encoder-decoder models pre-trained on a mixture of
unsupervised and supervised tasks.[^1] These models work well on a variety of
tasks by prepending task-specific prefixes to the input, e.g.:
`translate English to German: …`, `summarize: ….`, etc.

## Setup

Download and convert the model:

```sh
python convert.py --model <model>
```

This will make the `<model>.npz` file which MLX can read.

The `<model>` can be any of the following:

| Model Name | Model Size  |
| ---------- | ----------
| t5-small   | 60 million  |
| t5-base    | 220 million |
| t5-large   | 770 million |
| t5-3b      | 3 billion   |
| t5-11b     | 11 billion  |

## Generate

To gneerate text with the model, use the `t5.py` script:

```sh
python t5.py --model t5-small --prompt "translate English to German: A tasty apple"
```

Should give the output: `Ein schmackhafter Apfel`

To see a list of options run:

```sh
python t5.py --help
```

[^1]: For more information on T5 see the [original paper](https://arxiv.org/abs/1910.10683)
   or the [Hugging Face page](https://huggingface.co/docs/transformers/model_doc/t5).
