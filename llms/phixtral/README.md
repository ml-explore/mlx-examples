# Phixtral

Phixtral is a Mixture of Experts (MoE) architecture inspired by
[Mixtral](../mixtral/README.md) but based on the architecture of the
2.7B parameter model Phi-2 by Microsoft.[^1]
It was first introduced by combining finetuned versions of Phi-2 by Maxime
Labone [mlabonne/phixtral-4x2](https://huggingface.co/mlabonne/phixtral-4x2_8)

### Setup

Install the dependencies:

```
pip install -r requirements.txt
```

### Run
```
python generate.py --model <model_path> --prompt "hello"
```
For example:

```
python generate.py --model mlabonne/phixtral-4x2_8 --prompt "hello"
```
The `<model_path>` should be either a path to a local directory or a Hugging
Face repo with weights stored in `safetensors` format. If you use a repo from
the Hugging Face Hub, then the model will be downloaded and cached the first
time you run it. 

Run `python generate.py --help` to see all the options.

[^1]: For more details on the model see the [blog post](
https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/)
and the [Hugging Face repo](https://huggingface.co/microsoft/phi-2)
