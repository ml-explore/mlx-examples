# detectron2_example_mlx.py

import argparse
import torch
from torchvision.models.detection import detr
from torchvision.transforms import functional as F
from PIL import Image
import logging
import os

import mlx.core as mx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_detr_model():
    try:
        # Load the pre-trained DETR model from torchvision
        model = detr.detr_resnet50(pretrained=True)
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Error loading DETR model: {e}")
        raise

def process_image(image_path):
    try:
        # Load and preprocess the input image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        img = Image.open(image_path).convert("RGB")
        img_tensor = F.to_tensor(img).unsqueeze(0)
        return img_tensor
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise

def detect_objects_detr(model, img_tensor):
    try:
        # Perform object detection using DETR
        with torch.no_grad():
            outputs = model(img_tensor)
        return outputs
    except Exception as e:
        logger.error(f"Error during object detection: {e}")
        raise

def convert_outputs_to_detectron2(outputs):
    # Convert DETR outputs to Detectron2 format
    # This is a simple example, and you might need to adapt it based on the model's output structure
    detectron2_outputs = {
        "instances": {
            "pred_boxes": outputs["boxes"],
            "scores": outputs["scores"],
            "pred_classes": outputs["labels"],
        }
    }
    return detectron2_outputs

def visualize_results_detectron2(img, detectron2_outputs):
    try:
        # Visualize the detection results using Detectron2
        v = mx.visualization.Visualizer(img[0].cpu().numpy().transpose(1, 2, 0),
                                       metadata=mx.visualization.MetadataCatalog.get("coco"), scale=1.2)
        v = v.draw_instance_predictions(detectron2_outputs["instances"].to("cpu"))
        return Image.fromarray(v.get_image())
    except Exception as e:
        logger.error(f"Error visualizing results: {e}")
        raise

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Object Detection with DETR using Detectron2 for MLX")
        parser.add_argument("image_path", type=str, help="Path to the input image")
        parser.add_argument("--output", default="detr_output.png", help="Path to save the output image")
        args = parser.parse_args()

        # Load DETR model
        model = load_detr_model()

        # Process input image
        img_tensor = process_image(args.image_path)

        # Perform object detection using DETR
        outputs = detect_objects_detr(model, img_tensor)

        # Convert DETR outputs to Detectron2 format
        detectron2_outputs = convert_outputs_to_detectron2(outputs)

        # Visualize and save the results using Detectron2
        result_image = visualize_results_detectron2(img_tensor, detectron2_outputs)
        result_image.save(args.output)
        logger.info(f"Object detection results saved to {args.output}")

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
