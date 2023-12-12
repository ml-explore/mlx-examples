# CIFAR and ResNets

* This example shows how to run ResNets on CIFAR10 dataset, in accordance with the original [paper](https://arxiv.org/abs/1512.03385).
* Also illustrates how to use `mlx-data` to download and load the dataset.


## Pre-requisites
* Install the dependencies:

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
python main.py --cpu_only
```

For all available options, run:

```
python main.py --help
```
