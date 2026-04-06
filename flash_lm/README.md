# flash_lm 

Train an on-device LM with MLX

### Install

Install dependencies:

```
pip install -r requirements.tx
```

Install MLX on macOS:

```
pip install mlx
```

Or for CUDA:

```
pip install mlx[cuda13]
```

### Training

For pretraining:

```
python pretrian.py
```

For supervised fine-tuning (SFT):

```
python sft.py
```

### Generation

The model can be easily converted to a format compatible with `mlx_lm` for
generation.

Install `mlx-lm`:

```
pip install mlx-lm
```

Or for CUDA:

```
pip install mlx-lm[cuda13]
```

Then convert a given checkpoint:

```
python convert.py --checkpoint-dir path/to/checkpoint --save-dir path/to/mlx_lm_model
```

Then use any `mlx-lm` command or API:

```
mlx_lm.generate --model path/to/mlx_lm_model --prompt "Hi"
```

### Next Steps 

To customize the model change the default config (`configs/tiny.py`) or
make a new config and use it.

```
python pretrian.py --config my_custom_config.py
```
