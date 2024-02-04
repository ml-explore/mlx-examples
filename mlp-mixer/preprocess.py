import mlx.core as mx


def rescale(image: mx.array) -> mx.array:
    image = image * (1 / 255.0)
    return -1 + image * 2
