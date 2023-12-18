# T5

[T5](https://arxiv.org/pdf/1910.10683.pdf) are encoder-decoder models pre-trained on a multi-task mixture of unsupervised and supervised tasks. T5 works well on a variety of tasks out-of-the-box by prepending a different prefix to the input corresponding to each task, e.g.: `translate English to German: …`, `summarize: ….`

## Setup

Download and convert the model:

```sh
python convert.py --model t5-small
```

This will make the `{model}.npz` file which MLX can read.

## Generate

To run the model, use the `t5.py` script:

```sh
python t5.py --model t5-small --prompt "translate English to German: A tasty apple"
```

Should give the output: `Ein schmackhafter Apfel`

To see a list of options run:

```sh
python t5.py --help
```
