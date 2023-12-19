# Training a Vision Transformer on SpeechCommands

An example of training a Keyword Spotting Transformer[^1] on the Speech
Commands dataset[^2] with MLX. All supervised only configurations from the
paper are available.The example also illustrates how to use [MLX
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

After training with the `kwt1` architecture for 100 epochs, you
should see the following results:

```
Epoch: 99 | avg. Train loss 0.581 | avg. Train acc 0.826 | Throughput: 677.37 samples/sec
Epoch: 99 | Val acc 0.710
Testing best model from Epoch 98
Test acc -> 0.687
```

For the `kwt2` model, you should see:
```
Epoch: 99 | avg. Train loss 0.137 | avg. Train acc 0.956 | Throughput: 401.47 samples/sec
Epoch: 99 | Val acc 0.739
Testing best model from Epoch 97
Test acc -> 0.718
```

Note that this was run on an M1 Macbook Pro with 16GB RAM.

At the time of writing, `mlx` doesn't have built-in `cosine` learning rate
schedules, which is used along with the AdamW optimizer in the official
implementaiton. We intend to update this example once these features are added,
as well as with appropriate data augmentations.

[^1]: Based one the paper [Keyword Transformer: A Self-Attention Model for Keyword Spotting](https://www.isca-speech.org/archive/interspeech_2021/berg21_interspeech.html)
[^2]: We use version 0.02. See the [paper]((https://arxiv.org/abs/1804.03209) for more details.
