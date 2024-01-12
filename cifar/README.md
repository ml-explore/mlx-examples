# CIFAR and ResNets

An example of training a ResNet on CIFAR-10 with MLX. Several ResNet
configurations in accordance with the original
[paper](https://arxiv.org/abs/1512.03385) are available. The example also
illustrates how to use [MLX Data](https://github.com/ml-explore/mlx-data) to
load the dataset.

## Pre-requisites

Install the dependencies:

```
pip install -r requirements.txt
```

## Running the example

Run the example with:

```
python main.py
```

By default the example runs on the GPU. To run on the CPU, use: 

```
python main.py --cpu
```

For all available options, run:

```
python main.py --help
```

## Results

After training with the default `resnet20` architecture for 30 epochs, you
should see the following results:

```
Epoch: 29 | avg. Train loss 0.294 | avg. Train acc 0.897 | Throughput: 270.81 images/sec
Epoch: 29 | Test acc 0.841
```

Note this was run on an M1 Macbook Pro with 16GB RAM.

At the time of writing, `mlx` doesn't have built-in learning rate schedules.
We intend to update this example once these features are added.
