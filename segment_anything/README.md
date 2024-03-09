# Segment Anything

Segment Anything Model (SAM) in MLX. The implementation is ported from Meta AI Research's [Segment Anything](https://github.com/facebookresearch/segment-anything/tree/main).

## Installation
```bash
pip install -r requirements.txt
```

## Convert checkpoints

Download checkpoints from [Segment Anything repo](https://github.com/facebookresearch/segment-anything/tree/main?tab=readme-ov-file#model-checkpoints) and put them under `models/pt_models`

```bash
python scripts/convert_pt.py --pt-path=models/pt_models/[sam_vit_b_01ec64.pth | sam_vit_h_4b8939.pth | sam_vit_l_0b3195.pth]
```
The `safetensors` weights are generated under `models/mlx_models`

## Run

See examples `notebooks/predictor_example.ipynb` and `notebooks/automatic_mask_generator_example.ipynb` to try Segment Anything Model with MLX.

One can also generate masks from command line:
```bash
python scripts/amg.py --checkpoint <path/to/checkpoint> --model-type <model_type> --input <image_or_folder> --output <path/to/output>
```
