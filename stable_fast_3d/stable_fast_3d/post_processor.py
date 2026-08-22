"""
PixelShuffleUpsampleNetwork, the post_processor that turns the raw
1024-channel triplane into the final 40-channel, 4x-upsampled scene_codes
the decoder queries. Plain conv net - the only wrinkle is MLX has no
nn.PixelShuffle, so it's hand-rolled here.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

IN_CHANNELS = 1024
OUT_CHANNELS = 40
SCALE_FACTOR = 4
CONV_LAYERS = 4


def pixel_shuffle_nhwc(x: mx.array, r: int) -> mx.array:
    """NHWC equivalent of torch.nn.PixelShuffle: (B,H,W,C*r*r) -> (B,H*r,W*r,C).
    Torch's channel layout packs the flat C*r*r dim as (C,r,r) (i.e.
    index = c*r*r + r1*r + r2, verified against nn.PixelShuffle's own doc
    algorithm: view(*, C, r, r, H, W) -> permute(*, C, H, r, W, r))."""
    b, h, w, crr = x.shape
    c = crr // (r * r)
    x = x.reshape(b, h, w, c, r, r)
    x = x.transpose(0, 1, 4, 2, 5, 3)  # (B, H, r, W, r, C)
    return x.reshape(b, h * r, w * r, c)


class PixelShuffleUpsampleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        out_ch = OUT_CHANNELS * SCALE_FACTOR**2  # 640
        chans = [IN_CHANNELS, IN_CHANNELS, IN_CHANNELS, IN_CHANNELS, out_ch]
        self.convs = [
            nn.Conv2d(chans[i], chans[i + 1], kernel_size=3, padding=1)
            for i in range(CONV_LAYERS)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, H, W, IN_CHANNELS) NHWC
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i != len(self.convs) - 1:
                x = nn.relu(x)
        return pixel_shuffle_nhwc(x, SCALE_FACTOR)


def load_into(
    model: PixelShuffleUpsampleNetwork, weights: dict, prefix: str = "post_processor."
):
    # PyTorch Sequential indices: 0=Conv,1=ReLU,2=Conv,3=ReLU,4=Conv,5=ReLU,6=Conv,7=PixelShuffle
    conv_indices = [0, 2, 4, 6]
    for conv, idx in zip(model.convs, conv_indices):
        w = weights[f"{prefix}upsample.{idx}.weight"]  # (O,I,kh,kw)
        conv.weight = mx.transpose(w, (0, 2, 3, 1))  # -> (O,kh,kw,I)
        conv.bias = weights[f"{prefix}upsample.{idx}.bias"]
