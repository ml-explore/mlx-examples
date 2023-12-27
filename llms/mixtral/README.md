## Mixtral 8x7B

Run the Mixtral[^mixtral] 8x7B mixture-of-experts (MoE) model in MLX on Apple silicon.

This example also supports the instruction fine-tuned Mixtral model.[^instruct]

Note, for 16-bit precision this model needs a machine with substantial RAM (~100GB) to run.

### Setup

Install [Git Large File
Storage](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).
For example with Homebrew:

```
brew install git-lfs
```

Download the models from Hugging Face:

For the base model use:

```
export MIXTRAL_MODEL=Mixtral-8x7B-v0.1
```

For the instruction fine-tuned model use:

```
export MIXTRAL_MODEL=Mixtral-8x7B-Instruct-v0.1
```

Then run:

```
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/mistralai/${MIXTRAL_MODEL}/
cd $MIXTRAL_MODEL/ && \
  git lfs pull --include "consolidated.*.pt" && \
  git lfs pull --include "tokenizer.model"
```

Now from `mlx-exmaples/mixtral` convert and save the weights as NumPy arrays so
MLX can read them:

```
python convert.py --torch-path $MIXTRAL_MODEL/
```

To generate a 4-bit quantized model, use ``-q``. For a full list of options:

```
python convert.py --help
```

By default, the conversion script will make the directory `mlx_model` and save
the converted `weights.npz`, `tokenizer.model`, and `config.json` there.


### Generate

As easy as:

```
python mixtral.py --model-path mlx_model
```

For more options including how to prompt the model, run:

```
python mixtral.py --help
```

For the Instruction model, make sure to follow the prompt format:

```
[INST] Instruction prompt [/INST]
```

[^mixtral]: Refer to Mistral's [blog post](https://mistral.ai/news/mixtral-of-experts/) and the [Hugging Face blog post](https://huggingface.co/blog/mixtral) for more details.
[^instruc]: Refer to the [Hugging Face repo](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) for more
details
