# BERT Embedding Models

This example allows you to download and run BERT-based embedding models. Most
RoBERTA models should also work. The embeddings models are wrapped in a
SentenceTransformers-like API, so they're natively compatible with the the
Massive Text Embedding Benchmark (MTEB).

## Setup

Install the dependencies:

```shell
pip install -r requirements.txt
```

## Run

You can test on a few MTEB tasks:

```
python test_mteb.py
```

To use an embedding model, just import `EmbeddingModel` and instantiate it
with a path to a compatible Hugging Face BERT model. Make sure you choose the
right kind of pooling (`cls` vs. `mean`) specified in the model's
documentation, or the model won't work as well.


```python
from embeddings import EmbeddingModel
model = EmbeddingModel("princeton-nlp/sup-simcse-roberta-base")
model.encode(["hey there!"])
```
