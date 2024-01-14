# Copyright © 2023-2024 Apple Inc.

from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


class Bijector:
    def forward_and_log_det(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        raise NotImplementedError

    def inverse_and_log_det(self, y: mx.array) -> Tuple[mx.array, mx.array]:
        raise NotImplementedError


class AffineBijector(Bijector):
    def __init__(self, shift_and_log_scale: mx.array):
        self.shift_and_log_scale = shift_and_log_scale

    def forward_and_log_det(self, x: mx.array):
        shift, log_scale = mx.split(self.shift_and_log_scale, 2, axis=-1)
        y = x * mx.exp(log_scale) + shift
        log_det = log_scale
        return y, log_det

    def inverse_and_log_det(self, y: mx.array):
        shift, log_scale = mx.split(self.shift_and_log_scale, 2, axis=-1)
        x = (y - shift) * mx.exp(-log_scale)
        log_det = -log_scale
        return x, log_det


class MaskedCoupling(Bijector):
    def __init__(self, mask: mx.array, conditioner: nn.Module, bijector: Bijector):
        """Coupling layer with masking and conditioner."""
        self.mask = mask
        self.conditioner = conditioner
        self.bijector = bijector

    def apply_mask(self, x: mx.array, func: callable):
        """Transforms masked indices of `x` conditioned on unmasked indices using `func`."""
        x_masked = mx.where(self.mask, 0.0, x)
        bijector_params = self.conditioner(x_masked)
        y, log_det = func(bijector_params)
        log_det = mx.where(self.mask, log_det, 0.0)
        y = mx.where(self.mask, y, x)
        return y, mx.sum(log_det, axis=-1)

    def forward_and_log_det(self, x: mx.array):
        """Transforms masked indices of `x` conditioned on unmasked indices using bijector."""
        return self.apply_mask(
            x, lambda params: self.bijector(params).forward_and_log_det(x)
        )

    def inverse_and_log_det(self, y: mx.array):
        """Transforms masked indices of `y` conditioned on unmasked indices using bijector."""
        return self.apply_mask(
            y, lambda params: self.bijector(params).inverse_and_log_det(y)
        )
