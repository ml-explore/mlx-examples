from typing import Any, Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder


class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Args:
            image_encoder (ImageEncoderViT): The backbone used to encode the
                image into image embeddings that allow for efficient mask prediction.
            prompt_encoder (PromptEncoder): Encodes various types of input prompts.
            mask_decoder (MaskDecoder): Predicts masks from the image embeddings
                and encoded prompts.
            pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
            pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.pixel_mean = mx.array(pixel_mean).reshape(1, 1, -1)
        self.pixel_std = mx.array(pixel_std).reshape(1, 1, -1)

    def __call__(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, mx.array]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Args:
            batched_input (list(dict)): A list over input images, each a
                dictionary with the following keys. A prompt key can be
                excluded if it is not present.
                'image': The image as a mlx tensor in HxWx3 format,
                    already transformed for input to the model.
                'original_size': (tuple(int, int)) The original size of
                    the image before transformation, as (H, W).
                'point_coords': (mx.array) Batched point prompts for
                    this image, with shape BxNx2. Already transformed to the
                    input frame of the model.
                'point_labels': (mx.array) Batched labels for point prompts,
                    with shape BxN.
                'boxes': (mx.array) Batched box inputs, with shape Bx4.
                    Already transformed to the input frame of the model.
                'mask_inputs': (mx.array) Batched mask inputs to the model,
                    in the form BxHxWx1.
            multimask_output (bool): Whether the model should predict multiple
                disambiguating masks, or return a single mask.

        Returns:
            (list(dict)): A list over input images, where each element is
                as dictionary with the following keys.
                'masks': (mx.array) Batched binary mask predictions,
                    with shape BxCxHxW, where B is the number of input prompts,
                    C is determined by multimask_output, and (H, W) is the
                    original size of the image.
                'iou_predictions': (mx.array) The model's predictions
                    of mask quality, in shape BxC.
                'low_res_logits': (mx.array) Low resolution logits with
                    shape BxCxHxW, where H=W=256. Can be passed as mask input
                    to subsequent iterations of prediction.
        """
        input_images = mx.stack(
            [self.preprocess(x["image"]) for x in batched_input], axis=0
        )
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding[None],
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-3:-1],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: mx.array,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> mx.array:
        """
        Remove padding and upscale masks to the original image size.

        Args:
            masks (mx.array): Batched masks from the mask_decoder,
                in BxHxWxC format.
            input_size (tuple(int, int)): The size of the image input to the
                model, in (H, W) format. Used to remove padding.
            original_size (tuple(int, int)): The original size of the image
                before resizing for input to the model, in (H, W) format.

        Returns:
            (mx.array): Batched masks in BxCxHxW format, where (H, W)
                is given by original_size.
        """
        scale_factor = (
            self.image_encoder.img_size / masks.shape[1],
            self.image_encoder.img_size / masks.shape[2],
        )
        masks = nn.Upsample(
            scale_factor=scale_factor, mode="linear", align_corners=False
        )(masks)
        masks = masks[:, : input_size[0], : input_size[1]]
        scale_factor = (
            original_size[0] / masks.shape[1],
            original_size[1] / masks.shape[2],
        )
        masks = nn.Upsample(
            scale_factor=scale_factor, mode="linear", align_corners=False
        )(masks)
        return masks

    def preprocess(self, x: mx.array) -> mx.array:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-3:-1]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w

        if x.ndim == 3:
            pad_width = [(0, padh), (0, padw), (0, 0)]
        elif x.ndim == 4:
            pad_width = [(0, 0), (0, padh), (0, padw), (0, 0)]
        else:
            raise Exception("x.ndim can only be 3 or 4.")

        x = mx.pad(x, pad_width)
        return x
