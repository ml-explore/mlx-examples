# Train a Keyword Spotting Transformer on Speech Commands

An example of training a Keyword Spotting Transformer[^1] on the Speech
Commands dataset[^2] with MLX. All supervised only configurations from the
paper are available. The example also illustrates how to use [MLX
Data](https://github.com/ml-explore/mlx-data) to load and process an audio
dataset.

## Pre-requisites

Follow the [installation
instructions](https://ml-explore.github.io/mlx-data/build/html/install.html)
for MLX Data.

Install the remaining python requirements:

```
pip install -r requirements.txt
```

## Running the example

Run the example with:

```
python main.py
```

By default the example runs on the GPU. To run it on the CPU, use:

```
python main.py --cpu
```

For all available options, run:

```
python main.py --help
```

## Results

After training with the `kwt1` architecture for 10 epochs, you
should see the following results:

```
Epoch: 9 | avg. Train loss 0.519 | avg. Train acc 0.857 | Throughput: 661.28 samples/sec
Epoch: 9 | Val acc 0.861 | Throughput: 2976.54 samples/sec
Testing best model from epoch 9
Test acc -> 0.841
```

For the `kwt2` model, you should see:
```
Epoch: 9 | avg. Train loss 0.374 | avg. Train acc 0.895 | Throughput: 395.26 samples/sec
Epoch: 9 | Val acc 0.879 | Throughput: 1542.44 samples/sec
Testing best model from epoch 9
Test acc -> 0.861
```

Note that this was run on an M1 Macbook Pro with 16GB RAM.

At the time of writing, `mlx` doesn't have built-in `cosine` learning rate
schedules, which is used along with the AdamW optimizer in the official
implementation. We intend to update this example once these features are added,
as well as with appropriate data augmentations.

[^1]: Based one the paper [Keyword Transformer: A Self-Attention Model for Keyword Spotting](https://www.isca-speech.org/archive/interspeech_2021/berg21_interspeech.html)
[^2]: We use version 0.02. See the [paper]((https://arxiv.org/abs/1804.03209) for more details.
