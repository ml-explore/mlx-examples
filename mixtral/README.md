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
git-lfs clone https://huggingface.co/someone13574/mixtral-8x7b-32kseqlen
```

After that's done, combine the files:
```
cd mixtral-8x7b-32kseqlen/
cat consolidated.00.pth-split0 consolidated.00.pth-split1 consolidated.00.pth-split2 consolidated.00.pth-split3 consolidated.00.pth-split4 consolidated.00.pth-split5 consolidated.00.pth-split6 consolidated.00.pth-split7 consolidated.00.pth-split8 consolidated.00.pth-split9 consolidated.00.pth-split10 > consolidated.00.pth
```

Now from `mlx-exmaples/mixtral` convert and save the weights as NumPy arrays so
MLX can read them:

```
python convert.py --model_path mixtral-8x7b-32kseqlen/
```

The conversion script will save the converted weights in the same location.

After that's done, if you want to clean some stuff up:

```
rm mixtral-8x7b-32kseqlen/*.pth*
```

### Generate

As easy as:

```
python mixtral.py --model_path mixtral-8x7b-32kseqlen/
```

[^mixtral]: Refer to Mistral's [blog post](https://mistral.ai/news/mixtral-of-experts/) for more details.
