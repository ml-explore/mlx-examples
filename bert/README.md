# BERT

An implementation of BERT [(Devlin, et al., 2019)](https://aclanthology.org/N19-1423/) within MLX.

## Setup 

Install the requirements:

```
pip install -r requirements.txt
```

Then convert the weights with:

```
python convert.py \
    --bert-model bert-base-uncased \
    --mlx-model weights/bert-base-uncased.npz
```

## Usage

To use the `Bert` model in your own code, you can load it with:

```python
import mlx.core as mx
from model import Bert, load_model

model, tokenizer = load_model(
    "bert-base-uncased",
    "weights/bert-base-uncased.npz")

batch = ["This is an example of BERT working on MLX."]
tokens = tokenizer(batch, return_tensors="np", padding=True)
tokens = {key: mx.array(v) for key, v in tokens.items()}

output, pooled = model(**tokens)
```

The `output` contains a `Batch x Tokens x Dims` tensor, representing a vector
for every input token. If you want to train anything at a **token-level**,
you'll want to use this.

The `pooled` contains a `Batch x Dims` tensor, which is the pooled
representation for each input. If you want to train a **classification**
model, you'll want to use this.


## Test

You can check the output for the default model (`bert-base-uncased`) matches the
Hugging Face version with:

```
python test.py
```
