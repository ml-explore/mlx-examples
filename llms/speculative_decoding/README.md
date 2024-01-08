# Speculative Decoding

This example implements speculative decoding with the T5 model for text
generation.[^1][^2] Speculative decoding uses a smaller draft model to propose
several tokens, and a larger model to decide which tokens to accept. The
distribution of the generated text is identical to what the larger model would
produce on its own, but with far fewer forward passes of the large model since
it can evaluate the draft tokens in parallel.

### Setup

First, install the requirements:

```
cd speculative_decoding
pip install -r requirements.txt
```

Then convert the model and the draft model. We'll use T5-XXL (11B parameters)
for the main model. Convert it with:

```
python convert.py --model t5-11b
```

We'll use T5-small for the draft model. Convert it with:

```
python convert.py --model t5-small
```

### Run

You can run with the default arguments:

```
python main.py
```

To see a full list of options use:
```
python main.py --help
```

### Notes

Speculative decoding works well when most of the tokens from the draft model
are accepted by the larger model. That's more likely to happen if the models
are trained on similar data.

One way to increase the chance of accepting a draft token is with the parameter
`--delta`. This parameter can be in the range $[0, 1]$. If it is $1$ then all
the draft tokens will be accepted by the model. If it is $0$, then only draft
tokens that match the original acceptance criterion are kept.[^1] Values
closer to $1$ increase the chance that a draft token is accepted.

Conversely, the fewer draft tokens accepted by the main model, the more
expensive speculative decoding is. You can use `--num-draft` to tune the number
of draft tokens per model evaluation to reduce the number of discarded
draft tokens. Decreasing `--num-draft` will decrease the number of discarded
draft tokens at the expense of more large model evaluations.

[^1]: See the paper [Fast Inference from Transformers via Speculative
Decoding](https://arxiv.org/abs/2211.17192)
[^2]: For more information on T5 see the [original paper](https://arxiv.org/abs/1910.10683)
   or the [Hugging Face page](https://huggingface.co/docs/transformers/model_doc/t5).
