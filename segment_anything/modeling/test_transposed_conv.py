import torch
import mlx.core as mx
import numpy as np
import torch.nn as nn

size = 128

x = np.random.normal(size=[1, 16, size, size]).astype(np.float32)
x_pt = torch.tensor(x)
conv_t = nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2, bias=False)
weight = conv_t.weight.detach().numpy()
y_pt = conv_t(x_pt).squeeze()
y_pt = mx.array(y_pt.detach().numpy())
print(y_pt.shape)

in_mx = mx.array(x.transpose(0, 2, 3, 1))
wt_mx = mx.array(weight.transpose(1, 2, 3, 0))
print(in_mx.shape, wt_mx.shape)

# in_mx = mx.random.normal([2, 4, 4, 16]).astype(mx.float32)
# wt_mx = mx.random.normal([1, 2, 2, 16]).astype(mx.float32)
y_mx = mx.conv_general(
    in_mx,
    wt_mx,
    stride=1,
    padding=1,
    kernel_dilation=1,
    input_dilation=2,
    groups=1,
    flip=True,
).squeeze()
print(y_mx.shape)

# print(y_pt)
# print(y_mx)
print((mx.abs(y_pt - y_mx) < 1e-3).all())
