# Phi-2

Phi-2 is a 2.7B parameter model released by Microsoft[^1] and trained on a mixture
of GPT-4 outputs and clean web-text. Its performance rivals
much, much stronger models.

## Setup 

Download and convert the model:

```sh 
python convert.py
```

which will make a file `weights.npz`.

## Generate 

To generate text with the default prompt:

```sh
python model.py
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
python model.py --prompt <your prompt here> --max_tokens <max_token>
```

[^1]: For more details on the model see the [blog post](
https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/)
and the [Hugging Face repo](https://huggingface.co/microsoft/phi-2)
