# CIFAR and ResNets

An example of training a ResNet on CIFAR-10 with MLX. Several ResNet configurations in accordance with the original [paper](https://arxiv.org/abs/1512.03385) are available. Also illustrates how to use `mlx-data` to download and load the dataset.


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


## Throughput

On the tested device (M1 Macbook Pro, 16GB RAM), I get the following throughput with a `batch_size=256`:
```
Epoch: 0 | avg. tr_loss 2.074 | avg. tr_acc 0.216 | Train Throughput: 415.39 images/sec
```

When training on just the CPU (with the `--cpu` argument), the throughput is significantly lower (almost 30x!):
```
Epoch: 0 | avg. tr_loss 2.074 | avg. tr_acc 0.216 | Train Throughput: 13.5 images/sec
```

## Results
After training for 100 epochs, the following results were observed:
```
Epoch: 99 | avg. tr_loss 0.320 | avg. tr_acc 0.888 | Train Throughput: 416.77 images/sec
Epoch: 99 | test_acc 0.807
```
At the time of writing, `mlx` doesn't have in-built `schedulers`, nor a `BatchNorm` layer. We'll revisit this example for exact reproduction once these features are added.