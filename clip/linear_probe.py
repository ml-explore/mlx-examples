# Mirror of the Linear Probe Evaluation Script
# from the official CLIP Repository.

import mlx.core as mx
import numpy as np
from image_processor import CLIPImageProcessor
from mlx.data.datasets import load_cifar10
from model import CLIPModel
from PIL import Image
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm


def get_cifar10(batch_size, root=None):
    tr = load_cifar10(root=root).batch(batch_size)
    test = load_cifar10(root=root, train=False).batch(batch_size)

    return tr, test


def get_features(model, image_proc, iter):
    all_features = []
    all_labels = []

    for batch in tqdm(iter):
        image, label = batch["image"], batch["label"]
        x = image_proc([Image.fromarray(im) for im in image])
        y = mx.array(label)

        image_embeds = model.get_image_features(x)
        mx.eval(image_embeds)

        all_features.append(image_embeds)
        all_labels.append(y)

    return mx.concatenate(all_features), mx.concatenate(all_labels)


if __name__ == "__main__":
    model = CLIPModel.from_pretrained("mlx_model")
    image_proc = CLIPImageProcessor.from_pretrained("mlx_model")

    train_iter, test_iter = get_cifar10(batch_size=256)
    train_features, train_labels = get_features(model, image_proc, train_iter)
    test_features, test_labels = get_features(model, image_proc, test_iter)

    # Perform logistic regression
    # NOTE: The value of C should be determined via a hyperparameter sweep
    # using a validation split
    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
    classifier.fit(train_features, train_labels)

    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(test_features)
    accuracy = (test_labels.squeeze() == predictions).mean().item() * 100
    print(f"Accuracy = {accuracy:.3f}")
