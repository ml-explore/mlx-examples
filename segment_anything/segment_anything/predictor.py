from typing import Optional, Tuple

import mlx.core as mx
import numpy as np

from .sam import Sam
from .utils.transforms import ResizeLongestSide


class SamPredictor:
    def __init__(
        self,
        sam_model: Sam,
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Args:
            sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__()
        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.vision_encoder.img_size)
        self.reset_image()

    def set_image(
        self,
        image: np.ndarray,
        image_format: str = "RGB",
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Args:
            image (np.ndarray): The image for calculating masks. Expects an
                image in HWC uint8 format, with pixel values in [0, 255].
            image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        self.reset_image()
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        input_image = mx.array(input_image)[None, :, :, :]

        self.original_size = image.shape[:2]
        self.input_size = input_image.shape[1:3]
        input_image = self.model.preprocess(input_image)
        self.features = self.model.vision_encoder(input_image)
        self.is_image_set = True

    def predict(
        self,
        point_coords: Optional[mx.array],
        point_labels: Optional[mx.array],
        box: Optional[mx.array] = None,
        mask_input: Optional[mx.array] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched mlx tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Args:
            point_coords (mx.array or None): A BxNx2 array of point prompts to the
                model. Each point is in (X,Y) in pixels.
            point_labels (mx.array or None): A BxN array of labels for the
                point prompts. 1 indicates a foreground point and 0 indicates a
                background point.
            box (mx.array or None): A size 4 array giving a box prompt to the
                model, in XYXY format.
            mask_input (mx.array): A low resolution mask input to the model, typically
                coming from a previous prediction iteration. Has form BxHxWx1, where
                for SAM, H=W=256. Masks returned by a previous iteration of the
                predict method do not need further transformation.
            multimask_output (bool): If true, the model will return three masks.
                For ambiguous input prompts (such as a single click), this will often
                produce better masks than a single prediction. If only a single
                mask is needed, the model's predicted quality score can be used
                to select the best mask. For non-ambiguous prompts, such as multiple
                input prompts, multimask_output=False can give better results.
            return_logits (bool): If true, returns un-thresholded masks logits
                instead of a binary mask.

        Returns:
            (mx.array): The output masks in BxHxWxC format, where C is the
                number of masks, and (H, W) is the original image size.
            (mx.array): An array of shape BxC containing the model's
                predictions for the quality of each mask.
            (mx.array): An array of shape BxHxWxC, where C is the number
                of masks and H=W=256. These low res logits can be passed to
                a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        # Transform input prompts
        points = None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            points = (point_coords, point_labels)
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=box,
            masks=mask_input,
            pe_layer=self.model.shared_image_embedding,
        )

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.shared_image_embedding(
                self.model.prompt_encoder.image_embedding_size
            ),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(
            low_res_masks, self.input_size, self.original_size
        )

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks

    def get_image_embedding(self) -> mx.array:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert (
            self.features is not None
        ), "Features must exist if an image has been set."
        return self.features

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None
