# MNIST

This example shows how to train and run inference on MNIST models.

## Setup

Install the dependencies:

```
pip install -r requirements.txt
```

## Training

Train the model with:

```
python main.py --gpu
```

This will train a simple 2-layer MLP for 10 epochs and save the trained model to `model.safetensors`.

By default, the example runs on the CPU. To run on the GPU, use `--gpu`.

For a full list of options:

```
python main.py --help
```

## Inference

After training, you can run inference on the test set:

```bash
# Show predictions for 5 random samples
python infer.py --num-samples 5

# Interactive mode - visualize and predict specific samples
python infer.py --interactive

# Use a custom model path
python infer.py --model my_model.safetensors
```

The inference script provides:
- Predictions with confidence scores
- ASCII art visualization (no matplotlib required)
- Interactive mode to test specific samples

Example output:
```
✓ Sample 1234:
  True:       8
  Predicted:  8
  Confidence: 89.1%
  Top 3:
    8: 89.1%
    3: 5.2%
    9: 2.1%
```

## Other Frameworks

To run the PyTorch or JAX examples install the respective framework.
