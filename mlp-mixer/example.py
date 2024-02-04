import mlx.core as mx
import mlx.data as dx
from imagenet_classes import IMAGENET_CLASSES
from mixer import load
from preprocess import rescale

# Load model
model_name = "imagenet1k-MixerB-16"
model = load(model_name)

# Load and preprocess example images
dataset = (
    # Make a buffer (finite length container of samples) from the python list
    dx.buffer_from_vector(
        [
            {"image": b"assets/dog.jpeg"},
            {"image": b"assets/cat.jpeg"},
            {"image": b"assets/llama.jpeg"},
            {"image": b"assets/hamster.jpeg"},
            {"image": b"assets/elephant.jpeg"},
        ]
    )
    # Shuffle and transform to a stream
    .to_stream()
    # MLP Mixer preprocessing pipeline
    .load_image("image")
    .image_resize_smallest_side("image", 256)
    .image_center_crop("image", 224, 224)
    # Accumulate into batches
    .batch(5)
    .key_transform("image", rescale)
    # Finally, fetch batches in background threads
    .prefetch(prefetch_size=1, num_threads=1)
)
# Load example batch
[batch] = dataset
x = mx.array(batch["image"])
# Predict
logits = model(x)
classes = mx.argmax(logits, -1).tolist()
# Human readable labels
human_readable_classes = [IMAGENET_CLASSES[c] for c in classes]
print(human_readable_classes)
