# Copyright © 2023 Apple Inc.

import mlx.core as mx

from .config import DiffusionConfig


def _linspace(a, b, num):
    x = mx.arange(0, num) / (num - 1)
    return (b - a) * x + a


def _interp(y, x_new):
    """Interpolate the function defined by (arange(0, len(y)), y) at positions x_new."""
    x_low = x_new.astype(mx.int32)
    x_high = mx.minimum(x_low + 1, len(y) - 1)

    y_low = y[x_low]
    y_high = y[x_high]
    delta_x = x_new - x_low
    y_new = y_low * (1 - delta_x) + delta_x * y_high

    return y_new


class SimpleEulerSampler:
    """A simple Euler integrator that can be used to sample from our diffusion models.

    The method ``step()`` performs one Euler step from x_t to x_t_prev.
    """

    def __init__(self, config: DiffusionConfig):
        # Compute the noise schedule
        if config.beta_schedule == "linear":
            betas = _linspace(
                config.beta_start, config.beta_end, config.num_train_steps
            )
        elif config.beta_schedule == "scaled_linear":
            betas = _linspace(
                config.beta_start**0.5, config.beta_end**0.5, config.num_train_steps
            ).square()
        else:
            raise NotImplementedError(f"{config.beta_schedule} is not implemented.")

        alphas = 1 - betas
        alphas_cumprod = mx.cumprod(alphas)

        self._sigmas = mx.concatenate(
            [mx.zeros(1), ((1 - alphas_cumprod) / alphas_cumprod).sqrt()]
        )

    @property
    def max_time(self):
        return len(self._sigmas) - 1

    def sample_prior(self, shape, dtype=mx.float32, key=None):
        noise = mx.random.normal(shape, key=key)
        return (
            noise * self._sigmas[-1] * (self._sigmas[-1].square() + 1).rsqrt()
        ).astype(dtype)

    def add_noise(self, x, t, key=None):
        noise = mx.random.normal(x.shape, key=key)
        s = self.sigmas(t)
        return (x + noise * s) * (s.square() + 1).rsqrt()

    def sigmas(self, t):
        return _interp(self._sigmas, t)

    def timesteps(self, num_steps: int, start_time=None, dtype=mx.float32):
        start_time = start_time or (len(self._sigmas) - 1)
        assert 0 < start_time <= (len(self._sigmas) - 1)
        steps = _linspace(start_time, 0, num_steps + 1).astype(dtype)
        return list(zip(steps, steps[1:]))

    def step(self, eps_pred, x_t, t, t_prev):
        sigma = self.sigmas(t).astype(eps_pred.dtype)
        sigma_prev = self.sigmas(t_prev).astype(eps_pred.dtype)

        dt = sigma_prev - sigma
        x_t_prev = (sigma.square() + 1).sqrt() * x_t + eps_pred * dt

        x_t_prev = x_t_prev * (sigma_prev.square() + 1).rsqrt()

        return x_t_prev
