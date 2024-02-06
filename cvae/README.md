# Convolutional Variational Autoencoder (CVAE) on MNIST

Convolutional variational autoencoder (CVAE) implementation in MLX using
MNIST.[^1]

## Setup 

Install the requirements:

```
pip install -r requirements.txt
```

## Run


To train a VAE run:

```shell
python main.py
```

To see the supported options, do `python main.py -h`.

Training with the default options should give:

```shell
$ python train.py 
Options: 
  Device: GPU
  Seed: 0
  Batch size: 128
  Max number of filters: 64
  Number of epochs: 50
  Learning rate: 0.001
  Number of latent dimensions: 8
Number of trainable params: 0.1493 M
Epoch    0 | Loss   19200.76 | Throughput   413.73 im/s | Time    148.6 (s)
Epoch    1 | Loss   11817.73 | Throughput   412.47 im/s | Time    149.7 (s)
Epoch    2 | Loss   10835.43 | Throughput   414.91 im/s | Time    148.1 (s)
...
Epoch   47 | Loss    8320.16 | Throughput   420.39 im/s | Time    146.4 (s)
Epoch   48 | Loss    8310.72 | Throughput   418.78 im/s | Time    147.0 (s)
Epoch   49 | Loss    8307.03 | Throughput   420.12 im/s | Time    146.4 (s)
```

The throughput was measured on a 32GB M1 Max. 

Reconstructed and generated images will be saved after each epoch in the
`models/` path. Below are examples of reconstructed training set images and
generated images.

#### Reconstruction

![MNIST Reconstructions](assets/rec_mnist.png)

#### Generation 

![MNIST Samples](assets/samples_mnist.png)


## Limitations

At the time of writing, MLX does not have tranposed 2D convolutions. The
example approximates them with a combination of nearest neighbor upsampling and
regular convolutions, similar to the original U-Net. We intend to update this
example once transposed 2D convolutions are available.

[^1]: For a good overview of VAEs see the original paper [Auto-Encoding
  Variational Bayes](https://arxiv.org/abs/1312.6114) or [An Introduction to
  Variational Autoencoders](https://arxiv.org/abs/1906.02691).
