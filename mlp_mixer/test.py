import unittest

import jax.numpy as jnp
import mlx.core as mx
import mlx.data as dx
from mixer import MLX_WEIGHTS_PATH
from mixer import load as load_mlx
from mixer_jax import JAX_WEIGHTS_PATH
from mixer_jax import load as load_jax
from preprocess import rescale

TEST_JAX_WEIGHTS_PATH = JAX_WEIGHTS_PATH
TEST_MLX_WEIGHTS_PATH = MLX_WEIGHTS_PATH

TEST_MODELS = [
    "imagenet1k-MixerB-16",
    "imagenet1k-MixerL-16",
    "imagenet21k-MixerB-16",
    "imagenet21k-MixerL-16",
]


class TestMixer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_dataset = (
            # Make a buffer (finite length container of samples) from the python list
            dx.buffer_from_vector(
                [
                    {"image": b"assets/dog.jpeg"},
                    {"image": b"assets/cat.jpeg"},
                    {"image": b"assets/llama.jpeg"},
                    {"image": b"assets/hamster.jpeg"},
                    {"image": b"assets/elephant.jpeg"},
                ]
            )
            # Shuffle and transform to a stream
            .to_stream()
            # MLPMixer image pipeline
            .load_image("image")
            .image_resize_smallest_side("image", 256)
            .image_center_crop("image", 224, 224)
            # Accumulate into batches
            .batch(5)
            .key_transform("image", rescale)
            # Finally, fetch batches in background threads
            .prefetch(prefetch_size=1, num_threads=1)
        )
        # Load test batch
        [batch] = test_dataset
        cls.test_batch = batch["image"]

    def test_model_impl(self):
        for model in TEST_MODELS:
            print(f"testing {model}...")
            # Load MLX model
            mlx_model = load_mlx(model, TEST_MLX_WEIGHTS_PATH)
            # Load JAX model
            jax_model, jax_weights = load_jax(model, TEST_JAX_WEIGHTS_PATH)
            x = self.test_batch
            # Compare
            mlx_logits = jnp.array(mlx_model(mx.array(x)))
            jax_logits = jax_model.apply(jax_weights, jnp.array(x), train=False)
            mlx_labels = jnp.argmax(mlx_logits, axis=-1)
            jax_labels = jnp.argmax(jax_logits, axis=-1)
            assert jnp.allclose(mlx_labels, jax_labels)


if __name__ == "__main__":
    unittest.main()
