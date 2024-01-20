# Copyright © 2023-2024 Apple Inc.

from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from bijectors import AffineBijector, MaskedCoupling
from distributions import Normal


class MLP(nn.Module):
    def __init__(self, n_layers: int, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        layer_sizes = [d_in] + [d_hidden] * n_layers + [d_out]
        self.layers = [
            nn.Linear(idim, odim)
            for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

    def __call__(self, x):
        for l in self.layers[:-1]:
            x = nn.gelu(l(x))
        return self.layers[-1](x)


class RealNVP(nn.Module):
    def __init__(self, n_transforms: int, d_params: int, d_hidden: int, n_layers: int):
        super().__init__()

        # Alternating masks
        self.mask_list = [mx.arange(d_params) % 2 == i % 2 for i in range(n_transforms)]
        self.mask_list = [mask.astype(mx.bool_) for mask in self.mask_list]

        self.freeze(keys=["mask_list"])

        # Conditioning MLP
        self.conditioner_list = [
            MLP(n_layers, d_params, d_hidden, 2 * d_params) for _ in range(n_transforms)
        ]

        self.base_dist = Normal(mx.zeros(d_params), mx.ones(d_params))

    def log_prob(self, x: mx.array):
        """
        Flow back to the primal Gaussian and compute log-density,
        adding the transformation log-determinant along the way.
        """
        log_prob = mx.zeros(x.shape[0])
        for mask, conditioner in zip(self.mask_list[::-1], self.conditioner_list[::-1]):
            x, ldj = MaskedCoupling(
                mask, conditioner, AffineBijector
            ).inverse_and_log_det(x)
            log_prob += ldj
        return log_prob + self.base_dist.log_prob(x).sum(-1)

    def sample(
        self,
        sample_shape: Union[int, Tuple[int, ...]],
        key: Optional[mx.array] = None,
        n_transforms: Optional[int] = None,
    ):
        """
        Sample from the primal Gaussian and flow towards the target distribution.
        """
        x = self.base_dist.sample(sample_shape, key=key)
        for mask, conditioner in zip(
            self.mask_list[:n_transforms], self.conditioner_list[:n_transforms]
        ):
            x, _ = MaskedCoupling(
                mask, conditioner, AffineBijector
            ).forward_and_log_det(x)
        return x

    def __call__(self, x: mx.array):
        return self.log_prob(x)
