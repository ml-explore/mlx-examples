# Normalizing flow

Real NVP normalizing flow from [Dinh et al. (2016)](https://arxiv.org/abs/1605.08803) implemented using `mlx`. 

The example is written in a somewhat more object-oriented style than strictly necessary, with an eye towards extension to other use cases benefitting from arbitrary distributions and bijectors.

## Usage

The example can be run with
```
python main.py
```
which trains the normalizing flow on the two moons dataset and plots the result in `samples.png`. 

By default the example runs on the GPU. To run on the CPU, do 
```
python main.py --cpu
```

For all available options, run
```
python main.py --help
```

## Results

![Samples](./samples.png)
