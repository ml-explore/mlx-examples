# Speculative Decoding

This example implements [speculative decoding] for text generation.[^1].
Speculative decoding uses a smaller draft model to propose several tokens, and
then a larger model which decides which tokens to accept. The generated text is
identical to what the larger model would produce on its own, but with far fewer
forward passes of the large model since it can evaluate the draft tokens in
parallel.

### Setup

First, install the requirements:

```
cd speculative_decoding
pip install -r requirements.txt
```

### Run

You can run with the default arguments:

```
python main.py
```

Speculative decoding works well when most of the tokens from the draft model
are accepted by the larger model. That's more likely to happen if the models
are trained on similar data. The default setting in this example uses TinyLlama
as a draft morel for Llama 7B.

[^1] See the paper [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
