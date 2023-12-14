## Mixtral 8x7B

Run the Mixtral[^mixtral] 8x7B mixture-of-experts (MoE) model in MLX on Apple silicon.

Note, for 16-bit precision this model needs a machine with substantial RAM (~100GB) to run.

### Setup

Install [Git Large File
Storage](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).
For example with Homebrew:

```
brew install git-lfs
```

Download the models from Hugging Face:

```
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/
cd Mixtral-8x7B-v0.1/ && \
  git lfs pull --include "consolidated.*.pt" && \
  git lfs pull --include "tokenizer.model"
```

Now from `mlx-exmaples/mixtral` convert and save the weights as NumPy arrays so
MLX can read them:

```
python convert.py --model_path Mixtral-8x7B-v0.1/
```

The conversion script will save the converted weights in the same location.

### Generate

As easy as:

```
python mixtral.py --model_path Mixtral-8x7B-v0.1/
```

[^mixtral]: Refer to Mistral's [blog
  post](https://mistral.ai/news/mixtral-of-experts/) for more details.
