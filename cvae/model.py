import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

import utils


# from https://github.com/ml-explore/mlx-examples/blob/main/stable_diffusion/stable_diffusion/unet.py
def upsample_nearest(x, scale: int = 2):
    B, H, W, C = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, scale, W, scale, C))
    x = x.reshape(B, H * scale, W * scale, C)
    return x


class UpsamplingConv2d(nn.Module):
    """
    A convolutional layer that upsamples the input by a factor of 2. MLX does not yet
    support transposed convolutions, so we approximate them with nearest neighbor upsampling
    followed by a convolution. This is similar to the approach used in the original U-Net.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )

    def __call__(self, x):
        x = self.conv(upsample_nearest(x))
        return x


class Encoder(nn.Module):
    """
    A convolutional variational encoder. We do not map the input image
    deterministically to a latent vector. Instead, we map the input to
    a probability distribution in latent space and sample a latent vector
    fron that distribution. In this example, we linearly map the input
    image to a mean vector and a vector of standard deviations that
    parameterize a normal distribution.

    We can then sample from this distribution to generate a new image. Also,
    we can add an auxiliary loss to the network that forces the distribution
    to be close to a standard normal distribution. We use the KL divergence
    between the two distributions as this auxiliary loss.
    """

    def __init__(self, num_latent_dims, num_img_channels, max_num_filters):
        super().__init__()
        self.num_latent_dims = num_latent_dims
        self.num_img_channels = num_img_channels
        self.max_num_filters = max_num_filters
        # Track KL divergence. This is used as an auxiliary loss term.
        self.kl_div = mx.array(0.0)

        # we assume B x 64 x 64 x #img_channels input
        # Todo: add input shape attribute to the model to make it more flexible

        # HWC
        img_input_shape = (64, 64, num_img_channels)

        # number of filters in the convolutional layers
        num_filters_1 = max_num_filters // 4
        num_filters_2 = max_num_filters // 2
        num_filters_3 = max_num_filters

        # Output (BHWC):  B x 32 x 32 x num_filters_1
        self.conv1 = nn.Conv2d(num_img_channels, num_filters_1, 3, stride=2, padding=1)
        # Output (BHWC):  B x 16 x 16 x num_filters_2
        self.conv2 = nn.Conv2d(num_filters_1, num_filters_2, 3, stride=2, padding=1)
        # Output (BHWC):  B x 8 x 8 x num_filters_3
        self.conv3 = nn.Conv2d(num_filters_2, num_filters_3, 3, stride=2, padding=1)

        # Batch Normalization
        self.bn1 = nn.BatchNorm(num_filters_1)
        self.bn2 = nn.BatchNorm(num_filters_2)
        self.bn3 = nn.BatchNorm(num_filters_3)

        # Linear mappings to mean and standard deviation

        # std-dev is not directly outputted but rather as a
        # vector of log-variances. This is because the
        # standard deviation must be positive and the exp()
        # in __call__ ensures this. It might also be numerically
        # more stable.

        # divide the spatial dimensions by 8 because of the 3 strided convolutions
        output_shape = [num_filters_3] + [
            dimension // 8 for dimension in img_input_shape[:-1]
        ]

        flattened_dim = math.prod(output_shape)
        self.proj_mu = nn.Linear(flattened_dim, num_latent_dims)
        self.proj_log_var = nn.Linear(flattened_dim, num_latent_dims)

    def __call__(self, x):
        x = nn.leaky_relu(self.bn1(self.conv1(x)))
        x = nn.leaky_relu(self.bn2(self.conv2(x)))
        x = nn.leaky_relu(self.bn3(self.conv3(x)))
        x = mx.flatten(x, 1)  # flatten all dimensions except batch

        mu = self.proj_mu(x)
        logvar = self.proj_log_var(x)
        # Ensure this is the std deviation, not variance
        sigma = mx.exp(logvar * 0.5)

        # Generate a tensor of random values from a normal distribution
        eps = mx.random.normal(sigma.shape)

        # Perform the reparametrization step.
        # This allows us to backpropagate through it, which we could not do,
        # if we had just sampled from a normal distribution with mean mu and
        # standard deviation sigma. The multiplication with sigma and addition
        # of mu is just a linear transformation of the random values from the
        # normal distribution. The result is a random value from the distribution
        # with mean mu and standard deviation sigma. Backpropagation is possible
        # because the gradients of the random values are just 1 and the gradients
        # of the linear transformation are just the weights of the linear transformation.
        z = eps * sigma + mu

        # compute KL divergence
        # see Appendix B from VAE paper:    https://arxiv.org/abs/1312.6114
        self.kl_div = -0.5 * mx.sum(1 + logvar - mu * mu - logvar.exp())
        return z  # return latent vector


class Decoder(nn.Module):
    """A convolutional decoder"""

    def __init__(self, num_latent_dims, num_img_channels, max_num_filters):
        super().__init__()
        self.num_latent_dims = num_latent_dims
        self.num_img_channels = num_img_channels
        self.max_num_filters = max_num_filters
        self.input_shape = None

        # decoder layers
        num_filters_1 = max_num_filters
        num_filters_2 = max_num_filters // 2
        num_filters_3 = max_num_filters // 4

        # HWC
        img_output_shape = (64, 64, self.num_img_channels)

        # divide the last two dimensions by 8 because of the 3 upsampling convolutions
        self.input_shape = [dimension // 8 for dimension in img_output_shape[:-1]] + [
            num_filters_1
        ]
        flattened_dim = math.prod(self.input_shape)

        # Output: flattened_dim
        self.lin1 = nn.Linear(num_latent_dims, flattened_dim)
        # Output (BHWC):  B x 16 x 16 x num_filters_2
        self.upconv1 = UpsamplingConv2d(
            num_filters_1, num_filters_2, 3, stride=1, padding=1
        )
        # Output (BHWC):  B x 32 x 32 x num_filters_1
        self.upconv2 = UpsamplingConv2d(
            num_filters_2, num_filters_3, 3, stride=1, padding=1
        )
        # Output (BHWC):  B x 64 x 64 x #img_channels
        self.upconv3 = UpsamplingConv2d(
            num_filters_3, num_img_channels, 3, stride=1, padding=1
        )

        # Batch Normalizations
        self.bn1 = nn.BatchNorm(num_filters_2)
        self.bn2 = nn.BatchNorm(num_filters_3)

    def __call__(self, z):
        # unflatten the latent vector
        x = self.lin1(z)

        # reshape to BHWC
        x = x.reshape(
            -1, self.input_shape[0], self.input_shape[1], self.max_num_filters
        )

        # approximate transposed convolutions with nearest neighbor upsampling
        x = nn.leaky_relu(self.bn1(self.upconv1(x)))
        x = nn.leaky_relu(self.bn2(self.upconv2(x)))
        # sigmoid to ensure pixel values are in [0,1]
        x = mx.sigmoid(self.upconv3(x))
        return x


class CVAE(nn.Module):
    """A convolutional variational autoencoder consisting of an encoder and a decoder."""

    def __init__(self, num_latent_dims, num_img_channels, max_num_filters):
        super().__init__()
        self.num_latent_dims = num_latent_dims
        self.num_img_channels = num_img_channels
        self.max_num_filters = max_num_filters
        self.encoder = Encoder(num_latent_dims, num_img_channels, max_num_filters)
        self.decoder = Decoder(num_latent_dims, num_img_channels, max_num_filters)

    # forward pass of the data "x"
    def __call__(self, x):
        # image to latent vector
        z = self.encode(x)
        # latent vector to image
        x = self.decode(z)
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def get_kl_div(self):
        return self.encoder.kl_div

    def num_params(self):
        nparams = sum(x.size for k, x in tree_flatten(self.trainable_parameters()))
        return nparams

    def save(self, fname):
        # ensure the directory exists
        utils.ensure_folder_exists(fname)
        # save the model weights
        self.save_weights(fname)

    def load(self, fname):
        self.load_weights(fname)
