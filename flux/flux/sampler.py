from functools import lru_cache

import mlx.core as mx


class FluxSampler:
    def __init__(
        self, base_shift: float = 0.5, max_shift: float = 1.5, shift: bool = True
    ):
        self._base_shift = base_shift
        self._max_shift = max_shift
        self._shift = shift

    @lru_cache
    def timesteps(
        self, num_steps, image_sequence_length, start: float = 1, stop: float = 0
    ):
        t = mx.linspace(start, stop, num_steps + 1)

        if self._shift:
            x = image_sequence_length
            x1, x2 = 256, 4096
            y1, y2 = self._base_shift, self._max_shift
            mu = (x - x1) * (y2 - y1) / (x2 - x1) + y1
            t = mx.exp(mu) / (mx.exp(mu) + (1 / t - 1))

        return t.tolist()

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
