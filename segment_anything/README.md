# Segment Anything

An implementation of the Segment Anything Model (SAM) in MLX. See the original
repo by Meta AI for more details.[^1]

## Installation

```bash
pip install -r requirements.txt
```

## Convert

```bash
python convert.py --hf-path facebook/sam-vit-base
```

The `safetensor` weight file is downloaded from Hugging Face, converted, and
saved in the directory `models/mlx_models`.

The model sizes are:

- `facebook/sam-vit-base`
- `facebook/sam-vit-large`
- `facebook/sam-vit-huge`

## Run

See examples `notebooks/predictor_example.ipynb` and
`notebooks/automatic_mask_generator_example.ipynb` to try the Segment Anything
Model with MLX.

One can also generate masks from the command line:

```bash
python main.py --checkpoint <path/to/checkpoint> --model-type <model_type> --input <image_or_folder> --output <path/to/output>
```

[^1]: The original Segment Anything [GitHub repo](https://github.com/facebookresearch/segment-anything/tree/main).
