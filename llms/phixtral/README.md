# Phixtral

Phixtral is a Mixture of Experts (MoE) architecture inspired by
[Mixtral](../mixtral/README.md) but made by combinding fine-tuned versions of
Phi-2.[^1][^2]

### Setup

Install the dependencies:

```
pip install -r requirements.txt
```

### Run

```
python generate.py \
  --model mlabonne/phixtral-4x2_8 \
  --prompt "write a quick sort in Python"
```

Run `python generate.py --help` to see all the options.

[^1]: For more details on Phixtral, see the [Hugging Face repo](https://huggingface.co/mlabonne/phixtral-4x2_8).
[^2]: For more details on Phi-2 see Microsoft's [blog post](
https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/)
and the [Hugging Face repo](https://huggingface.co/microsoft/phi-2).
