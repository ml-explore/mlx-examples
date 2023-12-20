## Speculative Decoding with MLX

This example implements [speculative decoding](https://arxiv.org/abs/2211.17192), which allows you to use a smaller draft model to predict several tokens, and then a larger verification model to check them all in parallel. The results are output that is identical to what the larger model would produce, but with far fewer forward passes (as long as the reference model is good enough at guessing).

Install the requirements and then you can try it out:
```
cd speculative_decoding
pip install -r requirements.txt
python test.py
```

In order for that to happen, it's generally good if the models are trained on similar data, with a similar chat template, etc. For example, you could use Meta's 7B Llama as a draft model for the 13B Llama. In my tests, I've mostly used TinyLlama as a draft model for Llama-7B. The chat versions of TinyLlama and Llama-7B-Chat are trained with different templates, but it works OK. Alternatively, you can use base models, and a prompt to make the model act like a chat model (e.g. [URIAL](https://arxiv.org/abs/2312.01552)).

I believe the implementation is *correct* (it produces near-identical output with regular generation vs. speculative decoding, and when speculative decoding is enabled, the draft model does correctly predict many tokens). However, it assumes a batch size of 1 at the moment (I'm not actually sure how to handle batching where some drafts might have more correct tokens than others). Also I feel like it could be faster!

Before merging this in, I would appreciate some help understanding how to make this faster and optimizing the performance so it's actually useful!