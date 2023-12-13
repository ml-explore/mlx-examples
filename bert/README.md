# BERT

An implementation of BERT [(Devlin, et al., 2019)](https://aclanthology.org/N19-1423/) within MLX.

## Downloading and Converting Weights

The `convert.py` script relies on `transformers` to download the weights, and exports them as a single `.npz` file.

```
python convert.py \
    --bert-model bert-base-uncased \
    --mlx-model weights/bert-base-uncased.npz
```

## Usage

To use the `Bert` model in your own code, you can load it with:

```python
from model import Bert, load_model

model, tokenizer = load_model(
    "bert-base-uncased",
    "weights/bert-base-uncased.npz")

batch = ["This is an example of BERT working on MLX."]
tokens = tokenizer(batch, return_tensors="np", padding=True)
tokens = {key: mx.array(v) for key, v in tokens.items()}

output, pooled = model(**tokens)
```

The `output` contains a `Batch x Tokens x Dims` tensor, representing a vector for every input token.
If you want to train anything at a **token-level**, you'll want to use this.

The `pooled` contains a `Batch x Dims` tensor, which is the pooled representation for each input.
If you want to train a **classification** model, you'll want to use this.

## Comparison with 🤗 `transformers` Implementation

In order to run the model, and have it forward inference on a batch of examples:

```sh
python model.py \
  --bert-model bert-base-uncased \
  --mlx-model weights/bert-base-uncased.npz
```

Which will show the following outputs:
```
MLX BERT:
[[[-0.52508914 -0.1993871  -0.28210318 ... -0.61125606  0.19114694
    0.8227601 ]
  [-0.8783862  -0.37107834 -0.52238125 ... -0.5067165   1.0847603
    0.31066895]
  [-0.70010054 -0.5424497  -0.26593682 ... -0.2688697   0.38338926
    0.6557663 ]
  ...
```

They can be compared against the 🤗 implementation with:

```sh
python hf_model.py \
  --bert-model bert-base-uncased
```

Which will show:
```
 HF BERT:
[[[-0.52508944 -0.1993877  -0.28210333 ... -0.6112575   0.19114678
    0.8227603 ]
  [-0.878387   -0.371079   -0.522381   ... -0.50671494  1.0847601
    0.31066933]
  [-0.7001008  -0.5424504  -0.26593733 ... -0.26887015  0.38339025
    0.65576553]
  ...
```
