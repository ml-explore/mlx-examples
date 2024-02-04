# Image classification using MLP Mixer

An example of image classification using Google's `MLPMixer` [^1] in MLX, trained on ImageNet [^3]. `MLPMixer` is a classification model based entirely on multi-layer perceptrons (MLPs) that are repeatedly applied across either spatial locations or feature channels.

## Setup

Install the dependencies:

```shell
pip install -r requirements.txt
```

Next, download the official model from Google Storage and convert it to MLX. The
default model is [imagenet1k-MixerB-16](https://storage.googleapis.com/mixer_models/imagenet1k/Mixer-B_16.npz).

```
python convert.py
```

The script will by default download the model to the directory ``weights/mlx``.

## Run

You can use `MLPMixer` to classify images. 

```python
import mlx.core as mx
import mlx.data as dx
from mixer import load
from preprocess import rescale
from imagenet_classes import IMAGENET_CLASSES

# Load model
model_name = 'imagenet1k-MixerB-16'
model = load(model_name)

# Load and preprocess example images
dataset = (
    # Make a buffer (finite length container of samples) from the python list
    dx.buffer_from_vector(
        [
            {'image': b'assets/dog.jpeg'},
            {'image': b'assets/cat.jpeg'},
            {'image': b'assets/llama.jpeg'},
            {'image': b'assets/hamster.jpeg'},
            {'image': b'assets/elephant.jpeg'},
        ]
    )
    # Shuffle and transform to a stream
    .to_stream()
    # MLP Mixer preprocessing pipeline
    .load_image('image')
    .image_resize_smallest_side('image', 256)
    .image_center_crop('image', 224, 224)
    # Accumulate into batches
    .batch(5)
    .key_transform('image', rescale)
    # Finally, fetch batches in background threads
    .prefetch(prefetch_size=1, num_threads=1)
)
# Load example batch
[batch] = dataset
x = mx.array(batch['image'])
# Predict
logits = model(x)
classes = mx.argmax(logits, -1).tolist()
# Human readable labels
human_readable_classes = [IMAGENET_CLASSES[c] for c in classes]
print(human_readable_classes)
```
The outputs are : `['English foxhound', 'Egyptian cat', 'llama', 'hamster', 'Indian elephant, Elephas maximus']`. Run the above example with `python example.py`.

This example re-implements minimal image preprocessing to reduce
dependencies. For additional preprocessing functionality, refer to the 
paper [^1] and the official implementation [^2].

MLX `MLPMixer` has been tested and works with the following checkpoints:

- [imagenet1k-MixerB-16](https://storage.googleapis.com/mixer_models/imagenet1k/Mixer-B_16.npz)
- [imagenet1k-MixerL-16](https://storage.googleapis.com/mixer_models/imagenet1k/Mixer-L_16.npz)
- [imagenet21k-MixerB-16](https://storage.googleapis.com/mixer_models/imagenet21k/Mixer-B_16.npz)
- [imagenet21k-MixerL-16](https://storage.googleapis.com/mixer_models/imagenet21k/Mixer-L_16.npz)

You can run the tests with:

```shell
python test.py
```

To test new models, update the `TEST_MODELS` in `test.py`.

## Attribution

- `assets/cat.jpeg` is "Cat" by London's, licensed under CC BY-SA 2.0.
- `assets/dog.jpeg` is "Happy Dog" by tedmurphy, licensed under CC BY 2.0.
- `assets/hamster.jpeg` is "Russian Dwarf Hamster" by cdrussorusso, licensed under CC BY 2.0.
- `assets/llama.jpeg` is "Llama 2" by nao-cha, licensed under CC BY-SA 2.0.
- `assets/elephant.jpeg` is "Baby Elephant Chester Zoo" by Paolo Camera, licensed under CC BY 2.0.

[^1]: Refer to the original paper [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/pdf/2105.01601.pdf)
[^2]: Refer to the [official implementation](https://github.com/google-research/vision_transformer)
[^3]: Refer to [ImageNet website](https://www.image-net.org/)