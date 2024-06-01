from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np

from .predictor import SamPredictor
from .sam import Sam
from .utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_mlx,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)


class SamAutomaticMaskGenerator:
    def __init__(
        self,
        model: Sam,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[mx.array]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
    ) -> None:
        """
        Using a SAM model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM with a ViT-H backbone.

        Arguments:
            model (Sam): The SAM model to use for mask prediction.
            points_per_side (int or None): The number of points to be sampled
                along one side of the image. The total number of points is
                points_per_side**2. If None, 'point_grids' must provide explicit
                point sampling.
            points_per_batch (int): Sets the number of points run simultaneously
                by the model. Higher numbers may be faster but use more GPU memory.
            pred_iou_thresh (float): A filtering threshold in [0,1], using the
                model's predicted mask quality.
            stability_score_thresh (float): A filtering threshold in [0,1], using
                the stability of the mask under changes to the cutoff used to binarize
                the model's mask predictions.
            stability_score_offset (float): The amount to shift the cutoff when
                calculated the stability score.
            box_nms_thresh (float): The box IoU cutoff used by non-maximal
                suppression to filter duplicate masks.
            crop_n_layers (int): If >0, mask prediction will be run again on
                crops of the image. Sets the number of layers to run, where each
                layer has 2**i_layer number of image crops.
            crop_nms_thresh (float): The box IoU cutoff used by non-maximal
                suppression to filter duplicate masks between different crops.
            crop_overlap_ratio (float): Sets the degree to which crops overlap.
                In the first crop layer, crops will overlap by this fraction of
                the image length. Later layers with more crops scale down this overlap.
            crop_n_points_downscale_factor (int): The number of points-per-side
                sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
            point_grids (list(mx.array) or None): A list over explicit grids
                of points used for sampling, normalized to [0,1]. The nth grid in the
                list is used in the nth crop layer. Exclusive with points_per_side.
            min_mask_region_area (int): If >0, postprocessing will be applied
                to remove disconnected regions and holes in masks with area smaller
                than min_mask_region_area. Requires opencv.
            output_mode (str): The form masks are returned in. Can be 'binary_mask',
                'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
                For large resolutions, 'binary_mask' may consume large amounts of
                memory.
        """

        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils  # type: ignore # noqa: F401

        if min_mask_region_area > 0:
            import cv2  # type: ignore # noqa: F401

        self.predictor = SamPredictor(model)
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode

    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generates masks for the given image.

        Arguments:
            image (np.ndarray): The image to generate masks for, in HWC uint8 format.

        Returns:
            list(dict(str, any)): A list over records for masks. Each record is
                a dict containing the following keys:
                segmentation (dict(str, any) or np.ndarray): The mask. If
                    output_mode='binary_mask', is an array of shape HW. Otherwise,
                    is a dictionary containing the RLE.
                bbox (list(float)): The box around the mask, in XYWH format.
                area (int): The area in pixels of the mask.
                predicted_iou (float): The model's own prediction of the mask's
                    quality. This is filtered by the pred_iou_thresh parameter.
                point_coords (list(list(float))): The point coordinates input
                    to the model to generate this mask.
                stability_score (float): A measure of the mask's quality. This
                    is filtered on using the stability_score_thresh parameter.
                crop_box (list(float)): The crop of the image used to generate
                    the mask, given in XYWH format.
        """

        # Generate masks
        mask_data = self._generate_masks(image)

        # Filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0:
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )

        # Encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [
                coco_encode_rle(rle) for rle in mask_data["rles"]
            ]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # Write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)

        return curr_anns

    def _generate_masks(self, image: np.ndarray) -> MaskData:
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        # Iterate over image crops
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
            data.cat(crop_data)

        # Remove duplicate masks between crops
        if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            keep_by_nms = non_max_supression(
                data["boxes"].astype(mx.float32),
                scores,
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)

        data.to_numpy()
        return data

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        # Crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        self.predictor.set_image(cropped_im)

        # Get points for this crop
        points_scale = mx.array(cropped_im_size[::-1])[None]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # Generate masks for this crop in batches
        data = MaskData()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(
                points, cropped_im_size, crop_box, orig_size
            )
            data.cat(batch_data)
            del batch_data
        self.predictor.reset_image()

        # Remove duplicates within this crop.
        keep_by_nms = non_max_supression(
            data["boxes"].astype(mx.float32),
            data["iou_preds"],
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = mx.array([crop_box for _ in range(len(data["rles"]))])
        return data

    def _process_batch(
        self,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        orig_h, orig_w = orig_size

        masks, iou_preds, _ = self.predictor.predict(
            points[:, None, :],
            mx.ones((points.shape[0], 1), dtype=mx.int64),
            multimask_output=True,
            return_logits=True,
        )
        masks = masks.transpose(0, 3, 1, 2)
        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=mx.repeat(points, masks.shape[1], axis=0),
        )
        del masks

        # Filter by predicted IoU
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)

        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"],
            self.predictor.model.mask_threshold,
            self.stability_score_offset,
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.predictor.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(
            data["boxes"], crop_box, [0, 0, orig_w, orig_h]
        )
        if not mx.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_mlx(data["masks"])
        del data["masks"]

        return data

    @staticmethod
    def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(mx.array(mask)[None])
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))
        scores = mx.array(scores)

        # Recalculate boxes and remove any new duplicates
        masks = mx.concatenate(new_masks, axis=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = non_max_supression(
            boxes.astype(mx.float32),
            scores,
            iou_threshold=nms_thresh,
        )
        # Only recalculate RLEs for masks that have changed
        for i_mask, keep in enumerate(keep_by_nms):
            if not keep:
                continue
            if scores[i_mask] == 0.0:
                mask_mlx = masks[i_mask][None]
                mask_data["rles"][i_mask] = mask_to_rle_mlx(mask_mlx)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data


def box_area(boxes: mx.array) -> mx.array:
    """
    Computes the area of a set of bounding boxes, which are specified by their
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (mx.array[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        mx.array[N]: the area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def batched_iou(boxes_a: mx.array, boxes_b: mx.array) -> mx.array:
    """Compute IoU for batched boxes.

    Args:
        boxes_a (mx.array): [..., [x1, y1, x2, y2]] sized Mx4
        boxes_b (mx.array): [..., [x1, y1, x2, y2]] sized Nx4

    Returns:
        mx.array: MxN
    """

    area_a = box_area(boxes_a)  # M
    area_b = box_area(boxes_b)  # N

    top_left = mx.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
    bottom_right = mx.minimum(boxes_a[:, None, 2:], boxes_b[:, 2:])

    area_inter = mx.prod(mx.clip(bottom_right - top_left, a_min=0, a_max=None), 2)

    return area_inter / (area_a[:, None] + area_b - area_inter)


def non_max_supression(
    boxes: mx.array, scores: mx.array, iou_threshold: float = 0.5
) -> mx.array:
    sort_index = mx.argsort(-scores)
    boxes = boxes[sort_index]

    n_boxes = boxes.shape[0]
    ious = batched_iou(boxes, boxes)
    ious -= mx.eye(n_boxes)

    ious = np.array(ious)
    keep = np.ones(n_boxes, dtype=np.bool_)
    for i, iou in enumerate(ious):
        if not keep[i]:
            continue

        condition = iou <= iou_threshold
        keep = keep & condition

    return sort_index[mx.array(np.where(keep)[0])]
