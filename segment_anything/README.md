# Segment Anything

Segment Anything Model (SAM) in MLX. The implementation is ported from Meta AI Research's [Segment Anything](https://github.com/facebookresearch/segment-anything/tree/main).

## Installation
```bash
pip install -r requirements.txt
```

## Convert checkpoints

Download checkpoints from [Segment Anything repo](https://github.com/facebookresearch/segment-anything/tree/main?tab=readme-ov-file#model-checkpoints) and put them under `models/pt_models`

```bash
python convert_pt.py --pt-path=models/pt_models/[sam_vit_b_01ec64.pth | sam_vit_h_4b8939.pth | sam_vit_l_0b3195.pth]
```
The `safetensors` weights are generated under `models/mlx_models`

## Get Started

Take `predictor_example.ipybn` as an example to try Segment Anything Model with MLX.