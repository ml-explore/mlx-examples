# Phi-2

Phi-2 is a 2.7B parameter language model released by Microsoft with
performance that rivals much larger models.[^1] It was trained on a mixture of
GPT-4 outputs and clean web text.

Phi-2 efficiently runs on Apple silicon devices with 8GB of memory in 16-bit
precision.

## Setup

Download and convert the model:

```sh
python convert.py
```

To generate a 4-bit quantized model use the `-q` flag:

```
python convert.py -q
```

By default, the conversion script will make the directory `mlx_model` and save
the converted `weights.npz`, and `config.json` there.

> [!TIP] Alternatively, you can also download a few converted checkpoints from
> the [MLX Community](https://huggingface.co/mlx-community) organization on
> Hugging Face and skip the conversion step.


## Generate

To generate text with the default prompt:

```sh
python phi2.py
```

Should give the output:

```
Answer: Mathematics is like a lighthouse that guides us through the darkness of
uncertainty. Just as a lighthouse emits a steady beam of light, mathematics
provides us with a clear path to navigate through complex problems. It
illuminates our understanding and helps us make sense of the world around us.

Exercise 2:
Compare and contrast the role of logic in mathematics and the role of a compass
in navigation.

Answer: Logic in mathematics is like a compass in navigation. It helps
```

To use your own prompt:

```sh
python phi2.py --prompt <your prompt here> --max-tokens <max_tokens_to_generate>
```

To see a list of options run:

```sh
python phi2.py --help
```

[^1]: For more details on the model see the [blog post](
https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/)
and the [Hugging Face repo](https://huggingface.co/microsoft/phi-2)
