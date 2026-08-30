import unittest

from stable_diffusion.unet import Transformer2D


class TestTransformer2D(unittest.TestCase):
    def test_group_norm_matches_diffusers_epsilon(self):
        transformer = Transformer2D(
            in_channels=32,
            model_dims=32,
            encoder_dims=32,
            num_heads=4,
        )

        self.assertEqual(transformer.norm.eps, 1e-6)


if __name__ == "__main__":
    unittest.main()
