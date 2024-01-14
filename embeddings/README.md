# BERT Embedding Models
This example allows you to download and run BERT-based embedding models. (Most RoBERTA models are also ~equivalent to BERT, so those should work too.) The embeddings models are wrapped in a SentenceTransformers-like API, so they're natively compatible with MTEB (the Massive Text Embedding Benchmark).

To test, navigate into the `embeddings` directory, and install dependencies:
```
cd embeddings
# mteb, beir and sentence-transformers
pip install -r requirements.txt
```

Then you can test on a few MTEB tasks:

```
python test_mteb.py
```

To use an embedding model, just import EmbeddingModel and instantiate it with a path to a compatible HuggingFace BERT model. Make sure you choose the right kind of pooling (cls vs. mean) specified in the model's documentation, or you will experience degraded performance.

```python
from embeddings import EmbeddingModel
model = EmbeddingModel("princeton-nlp/sup-simcse-roberta-base")
model.encode(["hey there!"])
```