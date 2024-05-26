import math
from typing import List, Tuple, Type, Union

import mlx.core as mx
import mlx.nn as nn

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Args:
            transformer_dim (int): the channel dimension of the transformer
            transformer (nn.Module): the transformer used to predict masks
            num_multimask_outputs (int): the number of masks to predict
                when disambiguating masks
            activation (nn.Module): the type of activation to use when
                upscaling masks
            iou_head_depth (int): the depth of the MLP used to predict
                mask quality
            iou_head_hidden_dim (int): the hidden dimension of the MLP
                used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.upscale_conv1 = ConvTranspose2d(
            transformer_dim,
            transformer_dim // 4,
            kernel_size=2,
            stride=2,
            padding=1,
        )
        self.upscale_layer_norm = LayerNorm2d(transformer_dim // 4)
        self.activation = activation()
        self.upscale_conv2 = ConvTranspose2d(
            transformer_dim // 4,
            transformer_dim // 8,
            kernel_size=2,
            stride=2,
            padding=1,
        )
        self.output_hypernetworks_mlps = [
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 1)
            for i in range(self.num_mask_tokens)
        ]

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth - 2,
        )

    def __call__(
        self,
        image_embeddings: mx.array,
        image_pe: mx.array,
        sparse_prompt_embeddings: mx.array,
        dense_prompt_embeddings: mx.array,
        multimask_output: bool,
    ) -> Tuple[mx.array, mx.array]:
        """
        Predict masks given image and prompt embeddings.

        Args:
            image_embeddings (mx.array): the embeddings from the image encoder
            image_pe (mx.array): positional encoding
            sparse_prompt_embeddings (mx.array): the embeddings of the points and boxes
            dense_prompt_embeddings (mx.array): the embeddings of the mask inputs
            multimask_output (bool): Whether to return multiple masks or a single
                mask.

        Returns:
            mx.array: batched predicted masks
            mx.array: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, :, :, mask_slice]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: mx.array,
        image_pe: mx.array,
        sparse_prompt_embeddings: mx.array,
        dense_prompt_embeddings: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Predicts masks. See '__call__' for more details."""
        # Concatenate output tokens
        output_tokens = mx.concatenate(
            [self.iou_token.weight, self.mask_tokens.weight], axis=0
        )
        output_tokens = mx.broadcast_to(
            output_tokens[None],
            [
                sparse_prompt_embeddings.shape[0],
                output_tokens.shape[0],
                output_tokens.shape[1],
            ],
        )
        tokens = mx.concatenate((output_tokens, sparse_prompt_embeddings), axis=1)

        # Expand per-image data in batch direction to be per-mask
        src = mx.repeat(image_embeddings, repeats=tokens.shape[0], axis=0)
        src = src + dense_prompt_embeddings
        b, h, w, c = src.shape

        # Run the transformer
        hs, src = self.transformer(src, image_pe, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.reshape(b, h, w, c)
        src = self.upscale_conv1(src)
        src = self.upscale_layer_norm(src)
        src = self.activation(src)
        src = self.upscale_conv2(src)
        upscaled_embedding = self.activation(src)
        hyper_in_list: List[mx.array] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = mx.stack(hyper_in_list, axis=1)
        b, h, w, c = upscaled_embedding.shape

        masks = (
            (hyper_in @ upscaled_embedding.reshape(b, h * w, c).transpose(0, 2, 1))
            .transpose(0, 2, 1)
            .reshape(b, h, w, -1)
        )

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        self.proj_out = nn.Linear(hidden_dim, output_dim)
        self.sigmoid_output = sigmoid_output

    def __call__(self, x):
        x = nn.relu(self.proj_in(x))
        for i, layer in enumerate(self.layers):
            x = nn.relu(layer(x))
        x = self.proj_out(x)
        if self.sigmoid_output:
            x = mx.sigmoid(x)
        return x


# TODO: Naive implem. Replace when mlx.nn support conv_transpose
class ConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        bias: bool = True,
    ):
        super().__init__()

        kernel_size, stride, padding = map(
            lambda x: (x, x) if isinstance(x, int) else x,
            (kernel_size, stride, padding),
        )
        scale = math.sqrt(1 / (in_channels * kernel_size[0] * kernel_size[1]))
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, *kernel_size, in_channels),
        )
        if bias:
            self.bias = mx.zeros((out_channels,))

        self.padding = padding
        self.stride = stride
        self.dilation = dilation

    def _extra_repr(self):
        return (
            f"{self.weight.shape[-1]}, {self.weight.shape[0]}, "
            f"kernel_size={self.weight.shape[1:2]}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"bias={'bias' in self}"
        )

    def __call__(self, x):
        y = mx.conv_general(
            x,
            self.weight,
            stride=1,
            padding=self.padding,
            kernel_dilation=self.dilation,
            input_dilation=self.stride,
            flip=True,
        )
        if "bias" in self:
            y = y + self.bias
        return y
