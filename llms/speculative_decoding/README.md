# Speculative Decoding

This example implements speculative decoding with the T5 model for text
generation.[^1] Speculative decoding uses a smaller draft model to propose
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

Then convert the model and the draft model. For example, you can convert th
T5 11B model with:

```
python convert.py --model t5-11b
```

And for the draft model, convert the T5 small model with:

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
`--delta`. This parameter can be in the range `[0, 1]`. If it is `1` then all
the draft tokens will be accepted by the model. If it is `0`, then only draft
tokens which match the original acceptance criterion kept.[^1] Values closer to
`1` increase the chance that a draft token is accepted.

Conversely, the fewer draft tokens accepted by the model, the more expensive
speculative decoding is. You can use `--draft` to tune the number of draft
tokens per model evaluation in order to reduce the number of discarded draft
tokens.

[^1] See the paper [Fast Inference from Transformers via Speculative
Decoding](https://arxiv.org/abs/2211.17192)
