from typing import Optional, Tuple, Type

import mlx.core as mx
import mlx.nn as nn

from .common import LayerNorm2d


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Args:
            embed_dim (int): The prompts' embedding dimension
            image_embedding_size (tuple(int, int)): The spatial size of the
                image embedding, as (H, W).
            input_image_size (int): The padded size of the image as input
                to the image encoder, as (H, W).
            mask_in_chans (int): The number of hidden channels used for
                encoding input masks.
            activation (nn.Module): The activation to use when encoding
                input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        self.point_embed = [
            nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)
        ]
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )
        self.mask_embed = MaskEmbed(embed_dim, mask_in_chans, activation)
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def _embed_points(
        self,
        points: mx.array,
        labels: mx.array,
        pad: bool,
        pe_layer: nn.Module,
    ) -> mx.array:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = mx.zeros((points.shape[0], 1, 2))
            padding_label = -mx.ones((labels.shape[0], 1))
            points = mx.concatenate([points, padding_point], axis=1)
            labels = mx.concatenate([labels, padding_label], axis=1)
        point_embedding = pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding = mx.where(
            labels[..., None] == -1,
            self.not_a_point_embed.weight[:, None],
            point_embedding,
        )
        point_embedding = mx.where(
            labels[..., None] == 0,
            point_embedding + self.point_embed[0].weight[:, None],
            point_embedding,
        )
        point_embedding = mx.where(
            labels[..., None] == 1,
            point_embedding + self.point_embed[1].weight[:, None],
            point_embedding,
        )
        return point_embedding

    def _embed_boxes(self, boxes: mx.array, pe_layer: nn.Module) -> mx.array:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embed[2].weight
        corner_embedding[:, 1, :] += self.point_embed[3].weight
        return corner_embedding

    def _embed_masks(self, masks: mx.array) -> mx.array:
        """Embeds mask inputs."""
        mask_embedding = self.mask_embed(masks)
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[mx.array, mx.array]],
        boxes: Optional[mx.array],
        masks: Optional[mx.array],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def __call__(
        self,
        points: Optional[Tuple[mx.array, mx.array]],
        boxes: Optional[mx.array],
        masks: Optional[mx.array],
        pe_layer: nn.Module,
    ) -> Tuple[mx.array, mx.array]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Args:
            points (tuple(mx.array, mx.array) or none): point coordinates
                and labels to embed
            boxes (mx.array or none): boxes to embed
            masks (mx.array or none): masks to embed
            pe_layer (PositionEmbeddingRandom): shared position embedding
                layer

        Returns:
            mx.array: sparse embeddings for the points and boxes, with shape
                BxNx(embed_dim), where N is determined by the number of input points
                and boxes.
            mx.array: dense embeddings for the masks, in the shape
                Bx(embed_H)x(embed_W)x(embed_dim)
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = mx.zeros((bs, 0, self.embed_dim))
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(
                coords, labels, pad=(boxes is None), pe_layer=pe_layer
            )
            sparse_embeddings = mx.concatenate(
                [sparse_embeddings, point_embeddings], axis=1
            )
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes, pe_layer=pe_layer)
            sparse_embeddings = mx.concatenate(
                [sparse_embeddings, box_embeddings], axis=1
            )

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = mx.broadcast_to(
                self.no_mask_embed.weight,
                shape=(
                    bs,
                    self.image_embedding_size[0],
                    self.image_embedding_size[1],
                    self.embed_dim,
                ),
            )

        return sparse_embeddings, dense_embeddings


class MaskEmbed(nn.Module):
    def __init__(self, embed_dim, mask_in_chans, activation):
        super().__init__()
        self.conv1 = nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2)
        self.layer_norm1 = LayerNorm2d(mask_in_chans // 4)
        self.conv2 = nn.Conv2d(
            mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2
        )
        self.layer_norm2 = LayerNorm2d(mask_in_chans)
        self.conv3 = nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1)
        self.activation = activation()

    def __call__(self, x):
        x = self.activation(self.layer_norm1(self.conv1(x)))
        x = self.activation(self.layer_norm2(self.conv2(x)))
        return self.conv3(x)


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.positional_embedding = scale * mx.random.normal((2, num_pos_feats))

    def _pe_encoding(self, coords: mx.array) -> mx.array:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_embedding
        coords = 2 * mx.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return mx.concatenate([mx.sin(coords), mx.cos(coords)], axis=-1)

    def __call__(self, size: Tuple[int, int]) -> mx.array:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        grid = mx.ones((h, w), dtype=mx.float32)
        y_embed = grid.cumsum(axis=0) - 0.5
        x_embed = grid.cumsum(axis=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(mx.stack([x_embed, y_embed], axis=-1))
        return pe  # HWC

    def forward_with_coords(
        self, coords_input: mx.array, image_size: Tuple[int, int]
    ) -> mx.array:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input * 1
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.astype(mx.float32))  # B x N x C
