from typing import Tuple, Optional, Union
import math

import mlx.core as mx


class Distribution:
    def __init__(self):
        pass

    def sample(self, sample_shape: Union[int, Tuple[int, ...]], key: Optional[mx.array] = None) -> mx.array:
        raise NotImplementedError

    def log_prob(self, x: mx.array) -> mx.array:
        raise NotImplementedError

    def sample_and_log_prob(self, sample_shape: Union[int, Tuple[int, ...]], key: Optional[mx.array] = None) -> Tuple[mx.array, mx.array]:
        raise NotImplementedError

    def __call__(self, sample_shape: Union[int, Tuple[int, ...]], key: Optional[mx.array] = None) -> mx.array:
        return self.log_prob(self.sample(sample_shape, key=key))


class Normal(Distribution):
    def __init__(self, mu: mx.array, sigma: mx.array):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def sample(self, sample_shape: Union[int, Tuple[int, ...]], key: Optional[mx.array] = None):
        return mx.random.normal(sample_shape, key=key) * self.sigma + self.mu

    def log_prob(self, x: mx.array):
        return -0.5 * math.log(2 * math.pi) - mx.log(self.sigma) - 0.5 * ((x - self.mu) / self.sigma) ** 2

    def sample_and_log_prob(self, sample_shape: Union[int, Tuple[int, ...]], key: Optional[mx.array] = None):
        x = self.sample(sample_shape, key=key)
        return x, self.log_prob(x)
