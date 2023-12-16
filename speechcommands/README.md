# Training a Vision Transformer on SpeechCommands

An example of training [Keyword Spotting Transformer](https://www.isca-speech.org/archive/interspeech_2021/berg21_interspeech.html), a variant of the Vision Transformer, on the [Speech Commands](https://arxiv.org/abs/1804.03209) (v0.02) dataset with MLX. All supervised only configurations from the paper are available.The example also
illustrates how to use [MLX Data](https://github.com/ml-explore/mlx-data) to
load and process an audio dataset.

## Pre-requisites

Install `mlx`

```
pip install mlx==0.0.5
```

At the time of writing, the SpeechCommands dataset is not yet a part of a `mlx-data` release. Install `mlx-data` from source from this [commit](https://github.com/ml-explore/mlx-data/commit/ae3431648b8e1594d63175a8f121d9873aeb9daa).

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

At the time of writing, `mlx` doesn't have built-in `cosine` learning rate schedules, which is used along with the AdamW optimizer in the official implementaiton. We intend to update this example once these features
are added, as well as with appropriate data augmentations.