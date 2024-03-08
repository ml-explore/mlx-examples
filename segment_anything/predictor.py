from typing import Optional, Tuple

import mlx.core as mx
import numpy as np

from segment_anything.modeling import Sam

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
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
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
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        input_image_mlx = mx.array(input_image)
        input_image_mlx = input_image_mlx[None, :, :, :]
        self.set_mlx_image(input_image_mlx, image.shape[:2])

    def set_mlx_image(
        self,
        transformed_image: mx.array,
        original_image_size: Tuple[int, ...],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Args:
            transformed_image (mx.array): The input image, with shape
                1xHxWx4, which has been transformed with ResizeLongestSide.
            original_image_size (tuple(int, int)): The size of the image
                before transformation, in (H, W) format.
        """
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[3] == 3
            and max(*transformed_image.shape[1:3]) == self.model.image_encoder.img_size
        ), f"set_mlx_image input must be BHWC with long side {self.model.image_encoder.img_size}."
        self.reset_image()

        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[1:3])
        input_image = self.model.preprocess(transformed_image)
        self.features = self.model.image_encoder(input_image)
        self.is_image_set = True

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
            point_coords (np.ndarray or None): A Nx2 array of point prompts to the
                model. Each point is in (X,Y) in pixels.
            point_labels (np.ndarray or None): A length N array of labels for the
                point prompts. 1 indicates a foreground point and 0 indicates a
                background point.
            box (np.ndarray or None): A length 4 array given a box prompt to the
                model, in XYXY format.
            mask_input (np.ndarray): A low resolution mask input to the model, typically
                coming from a previous prediction iteration. Has form HxWx1, where
                for SAM, H=W=256.
            multimask_output (bool): If true, the model will return three masks.
                For ambiguous input prompts (such as a single click), this will often
                produce better masks than a single prediction. If only a single
                mask is needed, the model's predicted quality score can be used
                to select the best mask. For non-ambiguous prompts, such as multiple
                input prompts, multimask_output=False can give better results.
            return_logits (bool): If true, returns un-thresholded masks logits
                instead of a binary mask.

        Returns:
            (np.ndarray): The output masks in CxHxW format, where C is the
                number of masks, and (H, W) is the original image size.
            (np.ndarray): An array of length C containing the model's
                predictions for the quality of each mask.
            (np.ndarray): An array of shape CxHxW, where C is the number
                of masks and H=W=256. These low resolution logits can be passed to
                a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        # Transform input prompts
        coords_mlx, labels_mlx, box_mlx, mask_input_mlx = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_mlx = mx.array(point_coords, dtype=mx.float32)
            labels_mlx = mx.array(point_labels, dtype=mx.int64)
            coords_mlx, labels_mlx = coords_mlx[None, :, :], labels_mlx[None, :]
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_mlx = mx.array(box, dtype=mx.float32)
            box_mlx = box_mlx[None, :]
        if mask_input is not None:
            mask_input_mlx = mx.array(mask_input, dtype=mx.float32)
            mask_input_mlx = mask_input_mlx[None, :, :, :]

        masks, iou_predictions, low_res_masks = self.predict_mlx(
            coords_mlx,
            labels_mlx,
            box_mlx,
            mask_input_mlx,
            multimask_output,
            return_logits=return_logits,
        )

        masks_np = np.array(masks[0])
        iou_predictions_np = np.array(iou_predictions[0])
        low_res_masks_np = np.array(low_res_masks[0])
        return masks_np, iou_predictions_np, low_res_masks_np

    def predict_mlx(
        self,
        point_coords: Optional[mx.array],
        point_labels: Optional[mx.array],
        boxes: Optional[mx.array] = None,
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
            boxes (np.ndarray or None): A Bx4 array given a box prompt to the
                model, in XYXY format.
            mask_input (np.ndarray): A low resolution mask input to the model, typically
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

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
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
