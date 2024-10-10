import math
from functools import lru_cache

import mlx.core as mx


class FluxSampler:
    def __init__(
        self, base_shift: float = 0.5, max_shift: float = 1.5, shift: bool = True
    ):
        self._base_shift = base_shift
        self._max_shift = max_shift
        self._shift = shift

    def _time_shift(self, x, t):
        x1, x2 = 256, 4096
        t1, t2 = self._base_shift, self._max_shift
        exp_mu = math.exp((x - x1) * (t2 - t1) / (x2 - x1) + t1)
        t = exp_mu / (exp_mu + (1 / t - 1))
        return t

    @lru_cache
    def timesteps(
        self, num_steps, image_sequence_length, start: float = 1, stop: float = 0
    ):
        t = mx.linspace(start, stop, num_steps + 1)

        if self._shift:
            t = self._time_shift(image_sequence_length, t)

        return t.tolist()

    def random_timesteps(self, B, L, dtype=mx.float32, key=None):
        t = mx.random.uniform(shape=(B,), dtype=dtype, key=key)

        if self._shift:
            t = self._time_shift(L, t)

        return t

    def sample_prior(self, shape, dtype=mx.float32, key=None):
        return mx.random.normal(shape, dtype=dtype, key=key)

    def add_noise(self, x, t, noise=None, key=None):
        noise = (
            noise
            if noise is not None
            else mx.random.normal(x.shape, dtype=x.dtype, key=key)
        )
        return x * (1 - t) + t * noise

    def step(self, pred, x_t, t, t_prev):
        return x_t + (t_prev - t) * pred
