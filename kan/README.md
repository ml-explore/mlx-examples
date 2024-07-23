# KAN: Kolmogorov–Arnold Networks implementation on MNIST with MLX

This code contains an example implementation of training a Kolmogorov–Arnold Network (KAN) on the MNIST dataset using the MLX framework. This example demonstrates how to configure and train the model using various command-line arguments for flexibility.

Based on the [paper](https://arxiv.org/pdf/2404.19756)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Arguments](#arguments)
  - [Examples](#examples)
- [Model Architecture](#model-architecture)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run this example, you need to have Python and the necessary libraries installed. Follow these steps to set up your environment:

1. Install the required packages:

```sh
pip install -r requirements.txt
```

## Usage

You can run the script `main.py` to train the KAN model on the MNIST dataset. The script supports various command-line arguments for configuration.

### Arguments

- `--cpu`: Use the Metal back-end.
- `--use-kan-convolution`: Use the Convolution KAN architecture. Will give a error because its not implemented yet.
- `--dataset`: The dataset to use (`mnist` or `fashion_mnist`). Default is `mnist`.
- `--num_layers`: Number of layers in the model. Default is `2`.
- `--in-features`: Number input features. Default is `28`.
- `--out-features`: Number output features. Default is `28`.
- `--num-classes`: Number of output classes. Default is `10`.
- `--hidden_dim`: Number of hidden units in each layer. Default is `64`.
- `--num_epochs`: Number of epochs to train. Default is `10`.
- `--batch_size`: Batch size for training. Default is `64`.
- `--learning_rate`: Learning rate for the optimizer. Default is `1e-3`.
- `--weight-decay`: Weight decay for the optimizer. Default is `1e-4`.
- `--eval-report-count`: Number of epochs to report validations / test accuracy values. Default is `10`.
- `--save-path`: Path with the model name where the trained KAN model will be saved. Default is `traned_kan_model.safetensors`.
- `--seed`: Random seed for reproducibility. Default is `0`.

### Examples

#### Find all Arguments wioth descriptions

```sh
python main.py --help
```

#### Basic Usage

Train the KAN model on the MNIST dataset with default settings:

```sh
python main.py --dataset mnist
```

#### Custom Configuration

Train the KAN model with a custom configuration:

```sh
python main.py --dataset fashion_mnist --num_layers 3 --hidden_dim 128 --num_epochs 20 --batch_size 128 --learning_rate 0.0005 --seed 42
```

#### Using GPU

Train the KAN model using the CPU backend:

```sh
python main.py --cpu --dataset mnist
```

## Model Architecture

The `KAN` (Kolmogorov–Arnold Networks) class defines the model architecture. The network consists of multiple `KANLinear` layers, each defined by the provided parameters. The number of layers and the hidden dimension size can be configured via command-line arguments.

### Example Model Initialization

```python
layers_hidden = [28 * 28] + [hidden_dim] * (num_layers - 1) + [num_classes]
model = KAN(layers_hidden)
```

### KAN Class

The `KAN` class initializes a sequence of `KANLinear` layers based on the provided hidden layers configuration. Each layer performs linear transformations with kernel attention mechanisms.

```python
class KAN(nn.Module):
    def __init__(self, layers_hidden, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, base_activation=nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]):
        super().__init__()
        self.layers = []
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features, out_features, grid_size, spline_order, scale_noise, scale_base, scale_spline, base_activation, grid_eps, grid_range
                )
            )
    def __call__(self, x, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return mx.add(*(
            layer.regularization_loss(regularize_activation, regularize_entropy) 
            for layer in self.layers
        ))
```

### KanConvolutional Class

The KanConvolutional class defines the convolutional model architecture. The network consists of multiple KANConv layers, each defined by the provided parameters. This class is used for models that require convolutional layers.

```python
class KanConvolutional(nn.Module):
    def __init__(self, layers_hidden, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, base_activation=nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]):
        super().__init__()
        self.layers = []
        for in_channels, out_channels in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANConv(
                    in_channels, out_channels, grid_size, spline_order, scale_noise, scale_base, scale_spline, base_activation, grid_eps, grid_range
                )
            )
    def __call__(self, x, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return mx.add(*(
            layer.regularization_loss(regularize_activation, regularize_entropy) 
            for layer in self.layers
        ))
```