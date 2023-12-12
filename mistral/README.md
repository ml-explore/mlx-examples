# Mistral 

An example of generating text with Mistral using MLX.

Mistral 7B is one of the top large language models in its size class. It is
also fully open source with a permissive license[^1].

### Setup

Install the dependencies:

```
pip install -r requirements.txt
```

Next, download the model and tokenizer:

```
curl -O https://files.mistral-7b-v0-1.mistral.ai/mistral-7B-v0.1.tar
tar -xf mistral-7B-v0.1.tar
```

Then, convert the weights with:

```
python convert.py
```

The conversion script will save the converted weights in the same location.

### Run

Once you've converted the weights to MLX format, you can generate text with
the Mistral model:

```
python mistral.py --prompt "It is a truth universally acknowledged,"  --temp 0
```

Run `python mistral.py --help` for more details.

[^1]: Refer to the [blog post](https://mistral.ai/news/announcing-mistral-7b/)
and [github repository](https://github.com/mistralai/mistral-src) for more
details.
