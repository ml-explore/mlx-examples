# Mistral 

An example of generating text with Mistral using MLX.

Mistral 7B is one of the top large language models in its size class. It is also fully open source with a permissive license[^1].

### Setup

Install the dependencies:

```
pip install -r requirements.txt
```

Next, download the model and tokenizer.

```
curl -O https://files.mistral-7b-v0-1.mistral.ai/mistral-7B-v0.1.tar
tar -xf mistral-7B-v0.1.tar
```

Then, convert the weights with:

```
python convert.py <path_to_torch_weights> mlx_mistral_weights.npz
```

### Run

Once you've converted the weights to MLX format, you can interact with the
Mistral model:

```
python mistral.py mlx_mistral.npz tokenizer.model "hello"
```

Run `python mistral.py --help` for more details.

[^1]: Refer to the [blog post](https://mistral.ai/news/announcing-mistral-7b/) and [github repository](https://github.com/mistralai/mistral-src) for more details.
