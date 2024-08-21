"""
Based on: https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
"""

import mlx.nn as nn

class ReLUSquared(nn.Module):
    """
    Applies the relu^2 activation introduced in https://arxiv.org/abs/2109.08668v2
    """

    def __call__(self, input):
        return nn.relu(input).square()

ACT2FN = {
    "gelu": nn.GELU,
    "mish": nn.Mish,
    "relu": nn.ReLU,
    "relu2": ReLUSquared,
    "relu6": nn.ReLU6,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
}

def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")
